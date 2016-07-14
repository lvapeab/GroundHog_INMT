#!/usr/bin/env python

import theano
import argparse
import cPickle
import traceback
import logging
import time
import sys
from collections import OrderedDict

import itertools
import numpy
import experiments.nmt
from termcolor import colored
from experiments.nmt import \
    RNNEncoderDecoder, \
    prototype_phrase_state, \
    parse_input
from experiments.nmt.numpy_compat import argpartition
from isles_utils import *
import copy

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

    def search(self, seq, n_samples, eos_id, fixed_words={}, isles=[], max_N=None, minlen=1, idx2word=None,
               verbose=False):
        """
        This method actually performs the beam search.

        :param seq: Input sequence.
        :param n_samples: Beam size
        :param eos_id: End-of-sentence index
        :param fixed_words: Dictionary of words fixed by the user: {position: word}
        :param isles: Isles fixed by the user. List of (isle_index, [words]) (Although isle_index is never used
        :param max_N: Maximum number of words to generate between two isles.
        :param minlen: Target sentence minimum length
        :param idx2word: Mapping between indices and words
        :param verbose: Be verbose
        :return: Array of final hypotheses and their associated cost
        """

        num_models = len(self.enc_decs)
        c = []
        for i in xrange(num_models):
            c.append(self.comp_repr[i](seq)[0])
        states = []

        for i in xrange(num_models):
            states.append(map(lambda x: x[None, :], self.comp_init_states[i](c[i])))
        dim = states[0][0].shape[1]
        num_levels = len(states[0])
        fin_trans = []
        fin_costs = []
        if isles != []:
            unfixed_isles = filter(lambda x: not is_sublist(x[1], fixed_words.values()), [isle for isle in isles])
        else:
            unfixed_isles = []
        trans = [[]]
        costs = [0.0]
        k = 0
        while k < (3 * len(seq)):
            if n_samples == 0:
                break
            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t: t[-1], trans))
                          if k > 0
                          else numpy.zeros(beam_size, dtype="int64"))

            log_probs = sum(numpy.log(self.comp_next_probs[i](c[i], k, last_words, *states[i])[0]) for i in
                            xrange(num_models)) / num_models

            # Adjust log probs according to search restrictions
            if len(fixed_words.keys()) == 0:
                max_fixed_pos = 0
            else:
                max_fixed_pos = max(fixed_words.keys())

            if k < minlen or len(unfixed_isles) > 0 or k < max_fixed_pos:
                log_probs[:, eos_id] = -numpy.inf
            # If the current position is fixed, we fix it.
            if k in fixed_words:  # This position is fixed by the user
                # We fix the word
                log_probs[:, :] = -numpy.inf
                log_probs[:, fixed_words[k]] = 0.

            if len(unfixed_isles) == 0 or k in fixed_words:  # There are no remaining isles. Regular decoding.
                # Find the best options by calling argpartition of flatten array
                next_costs = numpy.array(costs)[:, None] - log_probs
                flat_next_costs = next_costs.flatten()
                # Element index to partition by. The kth element will be in its final sorted position and
                #  all smaller elements will be moved before it and all larger elements behind it.
                #  The order all elements in the partitions is undefined.
                best_costs_indices = argpartition(flat_next_costs.flatten(), n_samples)[:n_samples]

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
                for i, (orig_idx, next_word, next_cost) in enumerate(zip(trans_indices, word_indices, costs)):
                    new_trans[i] = trans[orig_idx] + [next_word]
                    new_costs[i] = next_cost
                    for level in range(num_levels):
                        for j in xrange(num_models):
                            new_states[j][level][i] = states[j][level][orig_idx]
                    inputs[i] = next_word
                for i in xrange(num_models):
                    new_states[i] = self.comp_next_states[i](c[i], k, inputs, *new_states[i])
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
                        if new_costs[i] != numpy.inf:
                            fin_trans.append(new_trans[i])
                            fin_costs.append(new_costs[i])
                for i in xrange(num_models):
                    states[i] = map(lambda x: x[indices], new_states[i])

            else:  # We are in the middle of two isles
                # logger.debug("Position %d in the middle of two isles"% k)
                hyp_trans = [[]] * max_N
                hyp_costs = [[]] * max_N
                trans_ = copy.copy(trans)
                costs_ = copy.copy(costs)
                states_ = [states] * (max_N + 1)
                n_samples_ = 1
                for kk in range(max_N):
                    beam_size = len(trans)
                    last_words = (numpy.array(map(lambda t: t[-1], trans_))
                                  if k + kk > 0
                                  else numpy.zeros(beam_size, dtype="int64"))
                    log_probs = sum(numpy.log(self.comp_next_probs[i](c[i], k + kk, last_words, *states_[kk][i])[0])
                                    for i in xrange(num_models)) / num_models
                    # Adjust log probs according to search restrictions
                    log_probs[:, eos_id] = -numpy.inf
                    if k + kk in fixed_words:  # This position is fixed by the user
                        log_probs[:, :] = -numpy.inf
                        log_probs[:, fixed_words[k + kk]] = 0.
                    # Find the best options by calling argpartition of flatten array
                    next_costs = numpy.array(costs_)[:, None] - log_probs
                    flat_next_costs = next_costs.flatten()
                    best_costs_indices = argpartition(flat_next_costs.flatten(), n_samples_)[:n_samples_]

                    # Decypher flatten indices
                    voc_size = log_probs.shape[1]
                    trans_indices = best_costs_indices / voc_size
                    word_indices = best_costs_indices % voc_size
                    costs_ = flat_next_costs[best_costs_indices]

                    # Form a beam for the next iteration
                    new_trans = [[]] * n_samples_
                    new_costs = numpy.zeros(n_samples_)
                    new_states = []
                    for i in xrange(num_models):
                        new_states.append([numpy.zeros((n_samples_, dim), dtype="float32") for level
                                           in range(num_levels)])
                    inputs = numpy.zeros(n_samples_, dtype="int64")
                    for i, (orig_idx, next_word, next_cost) in enumerate(zip(trans_indices, word_indices, costs_)):
                        new_trans[i] = trans_[orig_idx] + [next_word]
                        new_costs[i] = next_cost
                        for level in range(num_levels):
                            for j in xrange(num_models):
                                new_states[j][level][i] = states_[kk][j][level][orig_idx]
                        inputs[i] = next_word
                    for i in xrange(num_models):
                        new_states[i] = self.comp_next_states[i](c[i], k + kk, inputs, *new_states[i])
                    trans_ = []
                    costs_ = []
                    indices_ = []
                    for i in range(n_samples_):
                        trans_.append(new_trans[i])
                        costs_.append(new_costs[i])
                        indices_.append(i)
                    states__ = []
                    for i in xrange(num_models):
                        states__.append(numpy.asarray(map(lambda x: x[indices_], new_states[i])))
                    states_[kk + 1] = states__  # numpy.asarray(map(lambda x: x[indices_], new_states))
                    hyp_costs[kk] = costs_
                    hyp_trans[kk] = trans_
                """
                #TODO: Multiple best hypotheses??
                hyp_costs = numpy.asarray(hyp_costs, dtype="float32")
                hyp_trans = numpy.asarray(hyp_trans)

                flat_next_costs_hyp = hyp_costs.flatten()
                best_costs_indices_hyp = numpy.argmax(flat_next_costs_hyp)
                best_costs = flat_next_costs_hyp[best_costs_indices_hyp]
                best_hyp = hyp_trans.flatten()[best_costs_indices_hyp]

                # Decypher flatten indices
                voc_size = hyp_costs.shape[1]
                bests_n_words = best_costs_indices / voc_size # Number of words that we added to the hypothesis
                best_beams    = best_costs_indices % voc_size    # Position in the beam of the hypotheses



                print "hyp_costs=", hyp_costs
                print "len(hyp_costs)=", len(hyp_costs)
                print "best_costs=", best_costs
                print "best_costs.shape=", best_costs.shape
                print "hyp_trans=", hyp_trans
                print "len(hyp_trans)=", len(hyp_trans)
                print "best_trans=", best_hyp
                print "best_trans.shape=", len(best_hyp)
                print "bests_n_words=", bests_n_words
                print "best_beams   =", best_beams
                """
                best_n_words = -1
                min_cost = numpy.inf
                best_hyp = []
                for n_words in range(len(hyp_costs)):
                    for beam_index in range(len(hyp_costs[n_words])):
                        normalized_cost = hyp_costs[n_words][beam_index] / (n_words + 1)
                        if normalized_cost < min_cost:
                            min_cost = normalized_cost
                            best_n_words = n_words
                            best_hyp = hyp_trans[n_words][beam_index]
                assert best_n_words > -1, "Error in the rescoring approach"

                trans = hyp_trans[best_n_words]
                costs = hyp_costs[best_n_words]

                #trans = [best_hyp]*n_samples #hyp_trans[best_n_words]
                #costs = [min_cost]*n_samples #hyp_costs[best_n_words]
                states = states_[best_n_words + 1]
                best_n_words += 1
                logger.log(2, "Generating %d words from position %d" % (best_n_words, k))
                # We fix the words of the next isle
                stop = False
                while not stop and len(unfixed_isles) > 0:
                    k_counter = k + best_n_words
                    next_isle = unfixed_isles[0][1]
                    isle_prefixes = [next_isle[:i + 1] for i in range(len(next_isle))]
                    hyp = map(lambda x: idx2word[x] if idx2word.get(x) is not None else "UNK", best_hyp)
                    logger.log(2, "Hypothesis:%s" % str([(hyp[i], i) for i in range(len(hyp))]))
                    logger.log(2, "Isle: %s\n" % str(
                        map(lambda x: idx2word[x] if idx2word.get(x) is not None else "UNK", next_isle)))
                    _, start_pos = subfinder(next_isle, best_hyp)
                    if start_pos > -1:  # Isle completely included in the partial hypothesis
                        k_counter = start_pos
                        logger.log(2, "Isle included in hypothesis (position %d)" % k_counter)
                    else:
                        for i in range(len(best_hyp)):
                            if any(map(lambda x: x == best_hyp[i:], isle_prefixes)):
                                # Isle overlaps with the hypothesis: Remove isle
                                k_counter = i
                                stop = True
                                logger.log(2, "Isle overlapped (position %d)" % k_counter)
                                break
                    if k_counter == k + best_n_words:
                        logger.log(2, "Isle not included nor overlapped")
                        stop = True
                    for word in next_isle:
                        if fixed_words.get(k_counter) is None:
                            fixed_words[k_counter] = word
                            logger.log(2, "\t > Word %s (%d) will go to position %d" % (
                                idx2word[word] if idx2word.get(word) is not None else "UNK", word, k_counter))
                        else:
                            logger.log(2, "\t > Can't put word %s (%d) in position %d because it is in fixed_words" % (
                                idx2word[word] if idx2word.get(word) is not None else "UNK", word, k_counter))
                        k_counter += 1
                    del unfixed_isles[0]
                    #break #TODO: Juntar tantas islas como pueda? o solo una?
                k += best_n_words - 1
            k += 1

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if n_samples < 150:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seq, n_samples * 2, fixed_words=fixed_words, isles=isles,
                                   max_N=max_N, minlen=minlen, verbose=verbose, idx2word=idx2word, eos_id=eos_id)
            else:
                logger.warning("Translation failed. Returning partial translation.")
                max_fixed = max(fixed_words.keys())
                for i in range(n_samples):
                    for p in range(k, max_fixed+1):
                        if fixed_words.get(p) is not None:
                            new_trans[i].append(fixed_words[p])
                            new_costs[i] += 0.0
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])

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



def kl(p, q):
    """
    D_{KL} (P||Q) = \sum_i P(i)log (P(i)/Q(i))
    :param p:
    :param q:
    :return:
    """
    return numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))


def smoothed_kl(p, q):
    # Additive smoothing
    p = (p - 1) / p.shape[0]
    q = (q - 1) / q.shape[0]
    return numpy.sum(p * numpy.log(p / q), 0)


def sample(lm_model, seq, n_samples, eos_id, fixed_words={}, max_N=-1, isles=[],
           sampler=None, beam_search=None, normalize=False,
           alpha=1, verbose=False, idx2word=None):
    if beam_search:
        sentences = []
        if fixed_words is None or fixed_words == {}:
            minlen = len(seq) / 2
        else:
            minlen = max(len(seq) / 2, max(fixed_words.keys()))

        trans, costs = beam_search.search(seq, n_samples, eos_id=eos_id, fixed_words=fixed_words, isles=isles,
                                          max_N=max_N, minlen=minlen, verbose=verbose,
                                          idx2word=idx2word)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
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
                if lm_model.word_indxs[values[k, sidx]] == '<eol>':
                    break
                sen.append(lm_model.word_indxs[values[k, sidx]])
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
    alignments[:,len(src_seq)-1,range(x.shape[1])] = 0.  # Put source <eos> score to 0.
    hard_alignments = numpy.argmax(alignments, axis=1)  # trg_len x num_examples

    return hard_alignments


def replace_unknown_words(src_word_seq, trg_seq, trg_word_seq,
                          hard_alignment, unk_id, excluded_indices,
                          heuristic=0, mapping=[]):

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
                if UNK_src in mapping:
                    new_trans_words.append(mapping[UNK_src])
                else:
                    new_trans_words.append(UNK_src)
            elif heuristic == 2:
                # Use t-table if the source word starts with a lowercase letter. Otherwise copy
                # Sometimes works better than other heuristics
                if UNK_src in mapping and UNK_src.decode('utf-8')[0].islower():
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
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--state",
                        required=True, help="State to use")
    parser.add_argument("--beam-search",
                        action="store_true", help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",
                        type=int, help="Beam size")
    parser.add_argument("--source",
                        help="File of source sentences")
    parser.add_argument("--trans",
                        help="File to save translations in")
    parser.add_argument("--normalize",
                        action="store_true", default=False,
                        help="Normalize log-prob with the word count")
    parser.add_argument("--prefix",
                        action="store_true", default=False,
                        help="Pos-edit only with prefixes (classic ITP)")
    parser.add_argument("--verbose",
                        type=int, default=5,
                        help="Verbosity level: 0: No verbose. 1: Show hypotheses. 2: Show partial hypotheses.")
    parser.add_argument("--max-n",
                        type=int, default=5, help="Maximum number of words generated between isles")
    parser.add_argument("--interactive",
                        default=False, action="store_true",
                        help="Interactive post-editing?")
    parser.add_argument("--replaceUnk",
                        default=True, action="store_true",
                        help="Interactive post-editing?")
    parser.add_argument("--mapping",
                        help="Top1 unigram mapping (Source to target)")
    parser.add_argument("--heuristic", type=int, default=0,
            help="0: copy, 1: Use dict, 2: Use dict only if lowercase. Used only if a mapping is given. Default is 0.")
    parser.add_argument("--color",
                        default=False, action="store_true",
                        help="Colored hypotheses?")
    parser.add_argument("--references",
                        help="Reference sentence (for computing WSR)")
    parser.add_argument("--save-original",
                        default=False, action="store_true",
                        help="Save translations before post edition?")
    parser.add_argument("--save-original-to",
                        help="Save original hypotheses to")
    parser.add_argument("--models", nargs='+', required=True,
                        help="path to the models")
    parser.add_argument("changes",
                        nargs="?", default="",
                        help="Changes to state")
    return parser.parse_args()


def main():
    args = parse_args()
    state = prototype_phrase_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']),
                        format=" %(msecs)d: %(message)s")
    if args.verbose == 0:
        logger.setLevel(level=logging.INFO)
    elif args.verbose == 1:
        logger.setLevel(level=logging.DEBUG)
    elif args.verbose == 2:
        logger.setLevel(level=args.verbose)
    logger.debug("State: \n %s" % str(state))
    num_models = len(args.models)
    logger.info("Using an ensemble of %d models" % num_models)
    rng = numpy.random.RandomState(state['seed'])
    enc_decs = []
    lm_models = []
    alignment_fns = []
    for i in xrange(num_models):
        enc_decs.append(RNNEncoderDecoder(state, rng, skip_init=True, compute_alignment=args.replaceUnk))
        enc_decs[i].build()
        lm_models.append(enc_decs[i].create_lm_model())
        lm_models[i].load(args.models[i])
        if args.replaceUnk:
            alignment_fns.append(theano.function(inputs=enc_decs[i].inputs,
                                                 outputs=[enc_decs[i].alignment], name="alignment_fn"))
    if args.mapping:
        with open(args.mapping, 'rb') as f:
            mapping = cPickle.load(f)
        logger.debug("Loaded mapping file from %s" % str(args.mapping))
        heuristic = args.heuristic
    else:
        heuristic = 0
        mapping = None
    logger.info("Replacing unkown words according to heuristic %d" % heuristic)

    indx_word = cPickle.load(open(state['word_indx'], 'rb'))
    sampler = None
    if args.beam_search:
        beam_search = BeamSearch(enc_decs)
        beam_search.compile()
    else:
        raise NotImplementedError
        # sampler = enc_dec.create_sampler(many_samples=True)
    idict_src = cPickle.load(open(state['indx_word'], 'r'))
    unk_id = state['unk_sym_target']
    word2index = cPickle.load(open(state['word_indx_trgt'], 'r'))
    indx2word_trg = cPickle.load(open(state['indx_word_target'], 'rb'))
    eos_id = state['null_sym_target']
    if args.source and args.trans:
        # Actually only beam search is currently supported here
        assert beam_search
        assert args.beam_size
        try:
            fsrc = open(args.source, 'r')
            ftrans = open(args.trans, 'w')
            logger.info("Storing corrected hypotheses into: %s" % str(args.trans))

            if args.save_original:
                logger.info("Storing original hypotheses into: %s" % str(args.save_original_to))
                ftrans_ori = open(args.save_original_to, 'w')

            if not args.interactive:
                assert args.references is not None, "Automatic mode requires a reference file!"
                ftrg = open(args.references, 'r')
                target_lines = ftrg.read().split('\n')
                if target_lines[-1] == '':
                    target_lines = target_lines[:-1]
            start_time = time.time()
            n_samples = args.beam_size
            total_cost = 0.0
            logging.info("Beam size: {}".format(n_samples))
            total_errors = 0
            total_words = 0
            total_chars = 0
            total_mouse_actions = 0
            max_N = args.max_n
            if args.interactive:
                logger.info("Interactive mode\n\n")
                for n_line, line in enumerate(fsrc):
                    errors_sentence = 0
                    seqin = line.strip()
                    src_seq, src_words = parse_input(state, indx_word, seqin, idx2word=idict_src)
                    hypothesis_number = 0
                    fixed_words_user = OrderedDict()
                    validated_hypothesis = False
                    stage = 0
                    isles = []
                    while not validated_hypothesis:
                        print ""
                        sentences, costs, _ = sample(lm_models[0], src_seq, n_samples, eos_id, max_N=max_N, isles=isles,
                                                     fixed_words=fixed_words_user, sampler=sampler,
                                                     beam_search=beam_search, normalize=args.normalize,
                                                     verbose=args.verbose, idx2word=indx2word_trg)
                        best = numpy.argmin(costs)
                        hypothesis = sentences[best].split()
                        if args.color:
                            print_hypothesis = map(lambda x: 'colored(\'' + x + '\', \'green\')'
                            if fixed_words_user.get(hypothesis.index(x)) is not None
                            else 'str(\'' + x + '\')', hypothesis)

                            print "Sentence %d. Hypothesis %d:" % (n_line, hypothesis_number)
                            print " ".join(map(eval, print_hypothesis))
                        else:
                            print "Sentence %d. Hypothesis %d: %s" % (n_line, hypothesis_number, " ".join(hypothesis))
                        while stage < 3:
                            # Stage 1: Select isles
                            if stage == 0:
                                try:
                                    action = int(raw_input('Select the action to perform: \n'
                                                           ' \t 0: Validate sentence. \n'
                                                           ' \t 1: Select the correct words. \n'
                                                           ))

                                except ValueError:
                                    print "Invalid format."
                                    action = -1
                                    pass
                                if action == 0:
                                    stage = 3
                                    print >> ftrans, hypothesis
                                    total_words += len(hypothesis)
                                    total_errors += errors_sentence
                                    validated_hypothesis = True
                                    break
                                elif action == 1:
                                    # Stage 2: Regular post-editing (by prefixes)
                                    stage = 2
                                    fixed_positions = raw_input(
                                        'Select the correct words (1 - %d). 0 means no correct words ' % len(
                                            hypothesis))
                                    fixed_positions = fixed_positions.split()
                                    fixed_positions = [int(word_pos) - 1 for word_pos in fixed_positions]
                                    errors_sentence += 1
                                    for pos in range(0, len(hypothesis)):
                                        if pos not in fixed_positions:
                                            fixed_words_user[pos] = word2index[hypothesis[pos]]
                                        hypothesis_number += 1
                                    break
                                else:
                                    print "Unknown action."

                            elif stage == 2:
                                try:
                                    action = int(raw_input('Select the action to perform: \n'
                                                           ' \t 0: Validate sentence. \n'
                                                           ' \t 1: Substitute a word (validating its prefix). \n'
                                                           ))
                                except ValueError:
                                    print "Invalid format."
                                    action = -1
                                    pass
                                if action == 0:
                                    stage = 3
                                    print >> ftrans, hypothesis
                                    total_words += len(hypothesis)
                                    total_errors += errors_sentence
                                    validated_hypothesis = True
                                    break
                                elif action == 1:
                                    correct_word = int(raw_input('Select word to correct (1 - %d).' % len(hypothesis)))
                                    errors_sentence += 1
                                    hypothesis_number += 1
                                    new_word = raw_input('Substitute %s by: ' % hypothesis[correct_word - 1])
                                    for i in range(0, correct_word - 2):
                                        fixed_words_user[i] = word2index[hypothesis[i]]
                                    fixed_words_user[correct_word - 1] = word2index[new_word]
                                    break
                                else:
                                    print "Unknown action."
            else:
                #  Automatic mode
                for n_line, line in enumerate(fsrc):
                    errors_sentence = 0
                    mouse_actions_sentence = 0
                    hypothesis_number = 0
                    unk_indices = []

                    seqin = line.strip()
                    src_seq, src_words = parse_input(state, indx_word, seqin, idx2word=idict_src)
                    src_words = src_words.split()

                    logger.debug("\n \n Processing sentence %d" % (n_line + 1))
                    logger.debug("Source: %s" % line[:-1])
                    logger.debug("Target: %s" % target_lines[n_line])
                    reference = target_lines[n_line].split()
                    # 0. Get a first hypothesis
                    sentences, costs, trans = sample(lm_models[0], src_seq, n_samples, eos_id, sampler=sampler,
                                                     beam_search=beam_search,
                                                     normalize=args.normalize, verbose=args.verbose,
                                                     idx2word=indx2word_trg)
                    hypothesis_number += 1
                    best = numpy.argmin(costs)
                    hypothesis = sentences[best].split()
                    trg_seq = trans[best]
                    if args.replaceUnk and unk_id in trg_seq:
                        hard_alignments = compute_alignment(src_seq, trg_seq, alignment_fns)
                        hypothesis = replace_unknown_words(src_words, trg_seq, hypothesis, hard_alignments, unk_id,
                                              unk_indices, heuristic=heuristic, mapping = mapping).split()

                    if args.save_original:
                        print >> ftrans_ori, " ".join(hypothesis)
                    logger.debug("Hypo_%d: %s" % (hypothesis_number, " ".join(hypothesis)))
                    if hypothesis == reference:
                        # If the sentence is correct, we  validate it
                        pass
                    else:
                        checked_index_r = 0
                        checked_index_h = 0
                        last_checked_index = 0
                        unk_words = []
                        fixed_words_user = OrderedDict()  # {pos: word}
                        old_isles = []
                        while checked_index_r < len(reference):
                            validated_prefix = []
                            # Stage 1: Isles selection
                            #   1. Select the multiple isles in the hypothesis.
                            if not args.prefix:
                                hypothesis_isles = find_isles(hypothesis, reference)[0]
                                isle_indices = [(index, map(lambda x: word2index[x] if word2index.get(x) is not None
                                                                                    else unk_id, word))
                                                for index, word in hypothesis_isles]
                                logger.debug("Isles: %s" % (str(hypothesis_isles)))
                                if reference == " ".join([" ".join(h_isle[1]) for h_isle in hypothesis_isles]).split():
                                    logger.debug("Isles validating the full hypothesis")
                                    hypothesis = " ".join([" ".join(h_isle[1]) for h_isle in hypothesis_isles]).split()
                                    break
                                # Count only for non selected isles
                                mouse_actions_sentence += compute_mouse_movements(isle_indices, old_isles, last_checked_index) # Isles of length 1 account for 1 mouse action
                            else:
                                isle_indices = []

                            # Stage 2: Regular post editing
                            # From left to right, we will correct the hypotheses, taking into account the isles info
                            # At each timestep, the user can make two operations:
                            # Insertion of a new word at the end of the hypothesis
                            # Substitution of a word by another
                            while checked_index_r < len(reference):  # We check all words in the reference
                                if checked_index_h >= len(hypothesis):
                                    # Insertions (at the end of the sentence)
                                    errors_sentence += 1
                                    mouse_actions_sentence += 1
                                    new_word = reference[checked_index_r]
                                    new_word_index = word2index[new_word] \
                                        if word2index.get(new_word) is not None else unk_id
                                    validated_prefix.append(new_word_index)
                                    fixed_words_user[checked_index_h] = new_word_index
                                    if word2index.get(new_word) is None:
                                        unk_words.append(new_word)
                                        unk_indices.append(checked_index_h)
                                    #else:
                                    #    isle_indices[-1][1].append(word2index[new_word])
                                    logger.debug(
                                        '"%s" to position %d (end-of-sentence)' % (str(new_word), checked_index_h))
                                    last_checked_index = checked_index_h
                                    break
                                elif hypothesis[checked_index_h] != reference[checked_index_r]:
                                    errors_sentence += 1
                                    mouse_actions_sentence += 1
                                    new_word = reference[checked_index_r]
                                    # Substitution
                                    new_word_index = word2index[new_word] \
                                        if word2index.get(new_word) is not None else unk_id
                                    fixed_words_user[checked_index_h] = new_word_index
                                    validated_prefix.append(new_word_index)
                                    if word2index.get(new_word) is None:
                                        if checked_index_h not in unk_indices:
                                            unk_words.append(new_word)
                                            unk_indices.append(checked_index_h)
                                    logger.debug('"%s" to position %d' % (str(new_word), checked_index_h))
                                    last_checked_index = checked_index_h

                                    break
                                else:
                                    # No errors
                                    new_word_index = word2index[hypothesis[checked_index_h]] \
                                        if word2index.get(hypothesis[checked_index_h]) is not None else unk_id
                                    fixed_words_user[checked_index_h] = new_word_index
                                    validated_prefix.append(new_word_index)
                                    checked_index_h += 1
                                    checked_index_r += 1
                                    last_checked_index = checked_index_h
                            old_isles = [isle[1] for isle in isle_indices]
                            old_isles.append(validated_prefix)

                            # Generate a new hypothesis
                            logger.debug("")
                            sentences, costs, trans = sample(lm_models[0], src_seq, n_samples, eos_id,
                                                             fixed_words=copy.copy(fixed_words_user), max_N=max_N,
                                                             isles=isle_indices, sampler=sampler,
                                                             beam_search=beam_search, normalize=args.normalize,
                                                             verbose=args.verbose, idx2word=indx2word_trg)
                            hypothesis_number += 1
                            best = numpy.argmin(costs)
                            hypothesis = sentences[best].split()
                            trg_seq = trans[best]
                            if len(unk_indices) > 0:  # If we added some UNK word
                                if len(hypothesis) < len(unk_indices):  # The full hypothesis will be made up UNK words:
                                    for i, index in enumerate(range(0, len(hypothesis))):
                                        hypothesis[index] = unk_words[unk_indices[i]]
                                    for ii in range(i + 1, len(unk_words)):
                                        hypothesis.append(unk_words[ii])
                                else:  # We put each unknown word in the corresponding gap
                                    for i, index in enumerate(unk_indices):
                                        if index < len(hypothesis):
                                            hypothesis[index] = unk_words[i]
                                        else:
                                            hypothesis.append(unk_words[i])

                            hard_alignments = compute_alignment(src_seq, trg_seq, alignment_fns)
                            hypothesis = replace_unknown_words(src_words, trg_seq, hypothesis, hard_alignments, unk_id,
                                              unk_indices, heuristic=heuristic, mapping = mapping).split()
                            logger.debug("Target: %s" % target_lines[n_line])
                            logger.debug("Hypo_%d: %s" % (hypothesis_number, " ".join(hypothesis)))
                            if hypothesis == reference:
                                break
                        # Final check: The reference is a subset of the hypothesis: Cut the hypothesis
                        if len(reference) < len(hypothesis):
                            hypothesis = hypothesis[:len(reference)]
                            errors_sentence += 1
                            logger.debug("Cutting hypothesis")
                    """    
                    assert hypothesis == reference, "Error: The final hypothesis does not match with the reference! " \
                                                    "Sentence: %d \n" \
                                                    "Hypothesis: %s\n" \
                                                    "Reference:  %s" % (i, hypothesis, reference)
                    """
                    chars_sentence = sum(map(lambda x: len(x), hypothesis))
                    total_cost += costs[best]
                    total_errors += errors_sentence
                    total_words += len(hypothesis)
                    total_chars += chars_sentence
                    total_mouse_actions += mouse_actions_sentence + 1  # This +1 is the validation action
                    logger.debug("Final hypotesis: %s" % " ".join(hypothesis))
                    logger.debug("%d errors. "
                                 "Sentence WSR: %4f. "
                                 "Sentence mouse strokes: %d "
                                 "Sentence MAR: %4f. "
                                 "Sentence MAR_c: %4f. "
                                 "Accumulated WSR: %4f. "
                                 "Accumulated MAR: %4f. "
                                 "Accumulated MAR_c: %4f.\n\n\n\n\n\n" %
                                 (errors_sentence,
                                  float(errors_sentence) / len(hypothesis),
                                  mouse_actions_sentence + 1,
                                  float(mouse_actions_sentence + 1) / len(hypothesis),
                                  float(mouse_actions_sentence + 1) / chars_sentence,
                                  float(total_errors) / total_words,
                                  float(total_mouse_actions) / total_words,
                                  float(total_mouse_actions) / total_chars))

                    print >> ftrans, " ".join(hypothesis)

                    if (n_line + 1) % 50 == 0:
                        ftrans.flush()
                        if args.save_original:
                            ftrans_ori.flush()
                        logger.info("%d sentences processed" % (n_line + 1))
                        logger.info(
                                "Current speed is {} per sentence".format((time.time() - start_time) / (n_line + 1)))
                        logger.info("Current WSR is: %f" % (float(total_errors) / total_words))
                        logger.info("Current MAR is: %f" % (float(total_mouse_actions) / total_words))
                        logger.info("Current MAR_c is: %f" % (float(total_mouse_actions) / total_chars))

            print "Total number of errors:", total_errors
            print "Total number selections", total_mouse_actions
            print "WSR: %f" % (float(total_errors) / total_words)
            print "MAR: %f" % (float(total_mouse_actions) / total_words)
            print "MAR_c: %f" % (float(total_mouse_actions) / total_chars)

            print "Total cost of the translations: {}".format(total_cost)

            fsrc.close()
            ftrans.close()
            if args.save_original:
                ftrans_ori.close()
        except KeyboardInterrupt:
            print 'Interrupted!'
            print "Total number of corrections (up to now):", total_errors
            print "WSR: %f" % (float(total_errors) / total_words)
            print "SR: %f" % (float(total_mouse_actions) / n_line)

            sys.exit(0)
        except ValueError:
            pass
    else:
        while True:
            try:
                seqin = raw_input('Input Sequence: ')
                n_samples = int(raw_input('How many samples? '))
                alpha = None
                if not args.beam_search:
                    alpha = float(raw_input('Inverse Temperature? '))
                src_seq, src_words = parse_input(state, indx_word, seqin, idx2word=idict_src)
                print "Parsed Input:", src_words
            except Exception:
                print "Exception while parsing your input:"
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
