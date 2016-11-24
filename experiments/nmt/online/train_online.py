#!/usr/bin/env python

import argparse
import cPickle
import copy
import logging
import sys
import time
import traceback

import numpy
import theano

from experiments.nmt import \
    RNNEncoderDecoder, \
    prototype_phrase_state, \
    parse_input
from experiments.nmt.numpy_compat import argpartition
from groundhog.datasets.UnbufferedDataIterator import UnbufferedDataIterator
from experiments.nmt.online.online_utils import create_batch_from_seqs, loadSourceLanguageFromState, \
    loadTargetLanguageFromState
from groundhog.trainer.SGD_online import SGD as SGD
from groundhog.trainer.SGD_adadelta_online import SGD as Adadelta
from groundhog.trainer.SGD_adagrad import AdaGrad as AdaGrad
from groundhog.trainer.PassiveAggressive import PassiveAggressive

logger = logging.getLogger(__name__)

supported_algorithms = ["SGD", "AdaGrad", "PassiveAggressive", "Adadelta"]

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time


class BeamSearch(object):

    def __init__(self, enc_decs):
        self.enc_decs = enc_decs

    def compile(self):
        num_models = len(self.enc_decs)
        self.comp_repr = []
        self.comp_init_states = []
        self.comp_next_probs = []
        self.comp_next_states = []
        for i in xrange(num_models):
            self.comp_repr.append(self.enc_decs[i].create_representation_computer())
            self.comp_init_states.append(self.enc_decs[i].create_initializers())
            self.comp_next_probs.append(self.enc_decs[i].create_next_probs_computer())
            self.comp_next_states.append(self.enc_decs[i].create_next_states_computer())

    def search(self, seq, n_samples, eos_id, unk_id, ignore_unk=False, minlen=1):
        num_models = len(self.enc_decs)
        c = []
        for i in xrange(num_models):
            c.append(self.comp_repr[i](seq)[0])
        states = []
        for i in xrange(num_models):
            states.append(map(lambda x : x[None, :], self.comp_init_states[i](c[i])))
        dim = states[0][0].shape[1]

        num_levels = len(states[0])

        fin_trans = []
        fin_costs = []

        trans = [[]]
        costs = [0.0]

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break

            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t : t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))
            #log_probs = (numpy.log(self.comp_next_probs_0(c, k, last_words, *states)[0]) +  numpy.log(self.comp_next_probs_1(c, k, last_words, *states)[0]))/2.
            log_probs = sum(numpy.log(self.comp_next_probs[i](c[i], k, last_words, *states[i])[0]) for i in xrange(num_models))/num_models

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:,unk_id] = -numpy.inf
            if k < minlen:
                log_probs[:,eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples)[:n_samples]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size
            costs = flat_next_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = []
            for i in xrange(num_models):
                new_states.append([numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)])
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                for level in range(num_levels):
                    for j in xrange(num_models):
                        new_states[j][level][i] = states[j][level][orig_idx]
                inputs[i] = next_word
            for i in xrange(num_models):
                new_states[i]=self.comp_next_states[i](c[i], k, inputs, *new_states[i])

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != eos_id:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
            for i in xrange(num_models):
                states[i]=map(lambda x : x[indices], new_states[i])

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, eos_id=eos_id, unk_id=unk_id, ignore_unk=False, minlen=minlen)
            else:
                logger.warning("No appropriate translation: return empty translation")
                fin_trans=[[]]
                fin_costs = [0.0]

        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        return fin_trans, fin_costs

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def sample(lm_model, seq, n_samples, eos_id, sampler=None, beam_search=None, normalize=False,
           alpha=1, verbose=False, idx2word=None):
    if beam_search:
        sentences = []
        minlen = len(seq) / 2

        trans, costs = beam_search.search(seq, n_samples, eos_id=eos_id, unk_id=lm_model.target_language.unk_index,
                                           ignore_unk=False, minlen=minlen)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.target_language.indx_word, trans[i])
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose:
                logger.log(2, "{}: {}".format(costs[i], sentences[i]))
        return sentences, costs, trans

    elif sampler:
        sentences = []
        all_probs = []
        costs = []

        values, cond_probs = sampler(n_samples, 3 * (len(seq) - 1), alpha, seq)
        for sidx in xrange(n_samples):
            sen = []
            for k in xrange(values.shape[0]):
                if lm_model.target_language.indx_word[values[k, sidx]] == '<eol>':
                    break
                sen.append(lm_model.target_language.indx_word[values[k, sidx]])
            sentences.append(" ".join(sen))
            probs = numpy.array(cond_probs[:len(sen) + 1, sidx])
            all_probs.append(numpy.exp(-probs))
            costs.append(-numpy.sum(probs))
        if normalize:
            counts = [len(s.strip().split(" ")) for s in sentences]
            costs = [co / cn for co, cn in zip(costs, counts)]
        sprobs = numpy.argsort(costs)
        if verbose:
            for pidx in sprobs:
                logger.log(2, "Hypotheses {}: {} {} {}\n".format(pidx, -costs[pidx], all_probs[pidx], sentences[pidx]))

        return sentences, costs, None
    else:
        raise Exception("I don't know what to do")


def compute_alignment(src_seq, trg_seq, alignment_fns):
    num_models = len(alignment_fns)

    alignments = 0.
    x = numpy.asarray([src_seq], dtype="int64").T
    x_mask = numpy.ones((1, len(src_seq)), dtype="float32").T
    y = numpy.asarray([trg_seq], dtype="int64").T
    y_mask = numpy.ones((1, len(trg_seq)), dtype="float32").T
    for j in xrange(num_models):
        # target_len x source_len x num_examples
        alignments += numpy.asarray(alignment_fns[j](x, y, x_mask, y_mask)[0])
    alignments[:, len(src_seq) - 1,range(x.shape[1])] = 0.  # Put source <eos> score to 0.
    hard_alignments = numpy.argmax(alignments, axis=1)  # trg_len x num_examples

    return hard_alignments


def replace_unknown_words(src_word_seq, trg_seq, trg_word_seq,
                          hard_alignment, unk_id, excluded_indices,
                          heuristic=0, mapping=dict()):

    trans_words = trg_word_seq
    trans_seq = trg_seq
    hard_alignment = hard_alignment
    new_trans_words = []
    for j in xrange(len(trans_words)): # -1 : Don't write <eos>
        if trans_seq[j] == unk_id and j not in excluded_indices:
            UNK_src = src_word_seq[hard_alignment[j]]
            if heuristic == 0:  # Copy (ok when training with large vocabularies on en->fr, en->de)
                new_trans_words.append(UNK_src)
            elif heuristic == 1:
                # Use the most likely translation (with t-table). If not found, copy the source word.
                # Ok for small vocabulary (~30k) models
                if mapping.get(UNK_src) is not None:
                    new_trans_words.append(mapping[UNK_src])
                else:
                    new_trans_words.append(UNK_src)
            elif heuristic == 2:
                # Use t-table if the source word starts with a lowercase letter. Otherwise copy
                # Sometimes works better than other heuristics
                if mapping.get(UNK_src) is not None and UNK_src.decode('utf-8')[0].islower():
                    new_trans_words.append(mapping[UNK_src])
                else:
                    new_trans_words.append(UNK_src)
        else:
            new_trans_words.append(trans_words[j])
    to_write = ''
    for j, word in enumerate(new_trans_words):
        to_write = to_write + word
        if j < len(new_trans_words):
            to_write += ' '
    return to_write


def parse_args():
    parser = argparse.ArgumentParser("Retrain a translation model")
    parser.add_argument("--state", required=True, help="State to use")
    parser.add_argument("--algo", default=None, help="Online training algorithm. \n\t Supported algorithms: \n"
                                                     "\t \t AdaGrad\n"
                                                     "\t \t AdaDelta\n"
                                                     "\t \t SGD\n")
    parser.add_argument("--lr", type=float, default=0.1, help='Learning rate for the online algorithm (if necessary)')
    parser.add_argument("--wn", action="store_true", default=False, help='Apply weight noise in the online algorithm')
    parser.add_argument("--beam-search", action="store_true", help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size", type=int, help="Beam size")
    parser.add_argument("--source", help="File of source sentences")
    parser.add_argument("--trans", help="File to save translations in")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level: 0: No verbose. 1: Show hypotheses. 2: Show partial hypotheses.")
    parser.add_argument("--normalize", action="store_true", default=False, help="Normalize log-prob with the word count")
    parser.add_argument("--n-iters", type=int, default=1, help="Number of iterations for training the model. Default is 1.")
    parser.add_argument("--replaceUnk", default=False, action="store_true", help="Replace Unk")
    parser.add_argument("--mapping", help="Top1 unigram mapping (Source to target)")
    parser.add_argument("--heuristic", type=int, default=0,
                        help="0: copy, 1: Use dict, 2: Use dict only if lowercase. "
                             "Used only if a mapping is given. Default is 0.")
    parser.add_argument("--refs", help="Reference sentences")
    parser.add_argument("--models", nargs='+', required=True, help="path to the models")
    parser.add_argument("changes", nargs="?", default="", help="Changes to state")
    return parser.parse_args()


def main():
    args = parse_args()
    state = prototype_phrase_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=state['level'])
    if args.verbose == 0:
        logger.setLevel(level=logging.INFO)
    elif args.verbose == 1:
        logger.setLevel(level=logging.DEBUG)
    elif args.verbose == 2:
        logger.setLevel(level=args.verbose)

    num_sentences = 1
    # File reading
    fsrc = open(args.source, 'r')
    source_lines = [line for line in fsrc]
    logger.info("Storing hypotheses into: %s" % str(args.trans))
    # Some checks before loading the model and compiling the modules

    assert args.refs is not None, "Online learning mode requires a reference file!"
    ftrg = open(args.refs, 'r')
    target_lines = ftrg.read().split('\n')
    if target_lines[-1] == '':
        target_lines = target_lines[:-1]
    assert args.algo in supported_algorithms
    if args.algo is not None:
        assert args.algo in supported_algorithms
        algos = []
        batch_iters = []
    logger.debug("State: \n %s" % str(state))
    num_models = len(args.models)
    logger.info("Using an ensemble of %d models" % num_models)
    rng = numpy.random.RandomState(state['seed'])
    enc_decs = []
    lm_models = []
    alignment_fns = []
    probs_computers = []
    sourceLanguage = loadSourceLanguageFromState(state)
    targetLanguage = loadTargetLanguageFromState(state)
    # Model loading
    for i in xrange(num_models):
        enc_decs.append(RNNEncoderDecoder(state, rng, skip_init=True,
                                          compute_alignment=args.replaceUnk))
        enc_decs[i].build()
        lm_models.append(enc_decs[i].create_lm_model())
        lm_models[i].load(args.models[i])
        if 'PassiveAggressive' in args.algo:
            probs_computers.append(enc_decs[i].create_probs_computer(return_alignment=False))
        logger.info('Loading model %d from %s'%(i, args.models[i]))
        if args.replaceUnk:
            alignment_fns.append(theano.function(inputs=enc_decs[i].inputs,
                                                 outputs=[enc_decs[i].alignment],
                                                 name="alignment_fn"))
    heuristic = -1
    if args.replaceUnk:
        if args.mapping:
            with open(args.mapping, 'rb') as f:
                mapping = cPickle.load(f)
            logger.debug("Loaded mapping file from %s" % str(args.mapping))
            heuristic = args.heuristic
        else:
            heuristic = 0
            mapping = None
        logger.info("Replacing unkown words according to heuristic %d" % heuristic)
    else:
        logger.info("Not replacing unkown words")
        mapping = None
    if heuristic > 0:
        assert mapping is not None, 'When using heuristic 1 or 2, a mapping should be provided'

    word2indx_src = sourceLanguage.word_indx
    indx2word_src = sourceLanguage.indx_word
    unk_id = state['unk_sym_target']
    word2indx_trg = targetLanguage.word_indx
    indx2word_trg = targetLanguage.indx_word
    eos_id = state['null_sym_target']

    sampler = None
    if args.beam_search:
        logger.info('Creating beam search')
        beam_search = BeamSearch(enc_decs)
        beam_search.compile()
    else:
        raise NotImplementedError
    if args.algo is not None:
        state['lr'] = args.lr
        state['weight_noise'] = args.wn
        for i in xrange(num_models):
            batch_iters.append(UnbufferedDataIterator(args.source, args.refs, sourceLanguage, targetLanguage,
                                                      num_sentences, state['seqlen'], None))

            if 'PassiveAggressive' in args.algo:
                algos.append(eval(args.algo)(lm_models[i], state, batch_iters[i], probs_computers[i], enc_decs[i].predictions.word_probs))
            else:
                algos.append(eval(args.algo)(lm_models[i], state, batch_iters[i]))

    if args.source and args.trans:
        # Actually only beam search is currently supported here
        assert beam_search
        assert args.beam_size
        n_samples = args.beam_size
        logging.info("Beam size: {}".format(n_samples))
        for n_iter in range(1, args.n_iters + 1):
            logging.info('Starting iteration %d' %n_iter)
            ftrans = open(args.trans + '.iter_' + str(n_iter), 'w')
            try:
                start_time = time.time()
                total_cost = 0.0
                for n_line, line in enumerate(source_lines):
                    hypothesis_number = 0
                    unk_indices = []

                    seqin = line.strip()
                    src_seq, src_words = parse_input(state, word2indx_src, seqin, idx2word=indx2word_src)
                    src_words = seqin.split()

                    logger.debug("\n \n Processing sentence %d" % (n_line + 1))
                    logger.debug("Source: %s" % line[:-1])
                    logger.debug("Target: %s" % target_lines[n_line])
                    #reference = target_lines[n_line].split()
                    # 0. Get a first hypothesis
                    sentences, costs, trans = sample(lm_models[0], src_seq, n_samples, eos_id, sampler=sampler, verbose=args.verbose,
                                                     beam_search=beam_search, normalize=args.normalize, idx2word=indx2word_trg)
                    hypothesis_number += 1
                    best = numpy.argmin(costs)
                    hypothesis = sentences[best].split()
                    trg_seq = trans[best]
                    if args.replaceUnk and unk_id in trg_seq:
                        hard_alignments = compute_alignment(src_seq, trg_seq, alignment_fns)
                        hypothesis = replace_unknown_words(src_words, trg_seq, hypothesis, hard_alignments, unk_id,
                                              unk_indices, heuristic=heuristic, mapping=mapping).split()

                    print >> ftrans, " ".join(hypothesis)
                    logger.debug("Hypo_%d: %s" % (hypothesis_number, " ".join(hypothesis)))

                    # Online learning
                    if args.algo is not None:
                        #hypothesis_batch = create_batch_from_seqs(src_seq, hypothesis)
                        # Create batch
                        for i in xrange(num_models):
                            algos[i](" ".join(hypothesis))

            except KeyboardInterrupt:
                sys.exit(0)
            except ValueError:
                pass
            print "Total cost of the translations: {}".format(total_cost)
            ftrans.close()
            for i in xrange(num_models):
                batch_iters[i].reset()
        fsrc.close()
        ftrg.close()

if __name__ == "__main__":
    main()
