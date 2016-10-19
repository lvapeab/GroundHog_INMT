#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys
import theano
import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_phrase_state,\
    parse_input

from experiments.nmt.numpy_compat import argpartition

from collections import OrderedDict

logger = logging.getLogger(__name__)

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
            # TODO: report me in the paper!!!
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
    alignments[:,len(src_seq)-1,range(x.shape[1])] = 0.  # Put source <eos> score to 0.
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


def sample(lm_model, seq, n_samples, eos_id, unk_id,
        sampler=None, beam_search=None,
        ignore_unk=False, normalize=False,
        normalize_p = 1.0,
        alpha=1, verbose=False, wp=0.):
    if beam_search:
        sentences = []
        trans, costs = beam_search.search(seq, n_samples, eos_id=eos_id, unk_id=unk_id,
                ignore_unk=ignore_unk, minlen=len(seq) / 2)
        counts = [len(s) for s in trans]
        if normalize:
            costs = [co / ((max(cn,1))**normalize_p) + wp * cn for co, cn in zip(costs, counts)]
        else:
            costs = [co + wp * cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.target_language.indx_word, trans[i]) # Make sure that indices_to_words has been changed
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose:
                print "{}: {}".format(costs[i], sentences[i])
        return sentences, costs, trans
    elif sampler:
        raise NotImplementedError
    else:
        raise Exception("I don't know what to do")

def parse_args():
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-search) translations from a translation model")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--beam-search",
            action="store_true", help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",
            type=int, help="Beam size")
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--trans",
            help="File to save translations in")
    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--normalize-p",
            type=float, default=1.0,
            help="Controls preference to longer output. Only used if `normalize` is true.")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("--n-best", action="store_true", default=False,
            help="Write n-best list (of size --beam-size)")
    parser.add_argument("--start", type=int, default=0,
            help="For n-best, first sentence id")
    parser.add_argument("--wp", type=float, default=0.,
            help="Word penalty. >0: shorter translations \
                  <0: longer ones")
    parser.add_argument("--replaceUnk",
                        default=True, action="store_true",
                        help="Interactive post-editing?")
    parser.add_argument("--mapping",
                        help="Top1 unigram mapping (Source to target)")
    parser.add_argument("--heuristic", type=int, default=0,
            help="0: copy, 1: Use dict, 2: Use dict only if lowercase. Used only if a mapping is given. Default is 0.")
    parser.add_argument("--models", nargs = '+', required=True,
            help="path to the models")
    parser.add_argument("--changes",
            nargs="?", default="",
            help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_phrase_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    eos_id = state['null_sym_target']
    unk_id = state['unk_sym_target']

    num_models = len(args.models)
    logger.info("Using an ensemble of %d models" % num_models)
    rng = numpy.random.RandomState(state['seed'])
    enc_decs = []
    lm_models = []
    alignment_fns = []

    for i in xrange(num_models):
        enc_decs.append(RNNEncoderDecoder(state, rng, skip_init=True))
        enc_decs[i].build()
        lm_models.append(enc_decs[i].create_lm_model())
        lm_models[i].load(args.models[i])
        if args.replaceUnk:
            alignment_fns.append(theano.function(inputs=enc_decs[i].inputs,
                                                 outputs=[enc_decs[i].alignment],
                                                 name="alignment_fn"))
    indx_word = cPickle.load(open(state['word_indx'],'rb')) #Source w2i
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

    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_decs)
        beam_search.compile()
    else:
        raise NotImplementedError
        #sampler = enc_dec.create_sampler(many_samples=True)

    idict_src = cPickle.load(open(state['indx_word'],'r')) #Source i2w

    if args.source and args.trans:
        # Actually only beam search is currently supported here
        assert beam_search
        assert args.beam_size

        fsrc = open(args.source, 'r')
        ftrans = open(args.trans, 'w')

        n_samples = args.beam_size
        total_cost = 0.0
        logging.debug("Beam size: {}".format(n_samples))
        start_time = time.time()
        for i, line in enumerate(fsrc):
            seqin = line.strip()
            seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src) # seq is the ndarray of indices
            src_words = seqin.split()
            # For now, keep all input words in the model.
            # In the future, we may want to filter them to save on memory, but this isn't really much of an issue now
            if args.verbose:
                print "Parsed Input:", parsed_in
            sentences, costs, trans = sample(lm_models[0], seq, n_samples, sampler=sampler,
                    beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize,
                    normalize_p=args.normalize_p, eos_id=eos_id, unk_id=unk_id)

            if not args.n_best:
                best = numpy.argmin(costs)
                hypothesis = sentences[best].split()
                trg_seq = trans[best]
                if args.replaceUnk and unk_id in trg_seq:
                        hard_alignments = compute_alignment(seq, trg_seq, alignment_fns)
                        hypothesis = replace_unknown_words(src_words, trg_seq, hypothesis, hard_alignments, unk_id,
                                              [], heuristic=heuristic, mapping=mapping).split()
                print >> ftrans, " ".join(hypothesis)
            else:
                if args.replaceUnk and unk_id in trg_seq:
                    raise Exception, 'UNK replace not supported in N-best mode'
                order = numpy.argsort(costs)
                best = order[0]
                for elt in order:
                    print >>ftrans, str(i+args.start) + ' ||| ' + sentences[elt] + ' ||| ' + str(costs[elt])
            if args.verbose:
                print "Translation:", sentences[best]
            total_cost += costs[best]
            if (i + 1)  % 100 == 0:
                ftrans.flush()
                logger.debug("Current speed is {} per sentence".
                        format((time.time() - start_time) / (i + 1)))
        print "Total cost of the translations: {}".format(total_cost)
        print "Average time-per-sentence: {}".format((time.time()-start_time)/(i+1))
        fsrc.close()
        ftrans.close()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
