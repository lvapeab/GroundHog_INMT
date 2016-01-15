#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys
from collections import OrderedDict
import numpy
import experiments.nmt
from termcolor import colored
from experiments.nmt import \
    RNNEncoderDecoder, \
    prototype_phrase_state, \
    parse_input

from sklearn.externals import joblib
from experiments.nmt.numpy_compat import argpartition
from isles_utils import find_isles
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
    def __init__(self, enc_dec):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, n_samples, fixed_words= {}, isles = [], max_N = None,
               ignore_unk=False, minlen=1, verbose=False, idx2word=None):
        c = self.comp_repr(seq)[0]
        states = map(lambda x: x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]
        num_levels = len(states)
        fin_trans = []
        fin_costs = []
        inf = 1e3
        """
            k: Beam size
            trans: k-sized list of lists. Each sublist contains a translation hypothesis
            costs: k-sized list of log-probs. Each sublist contains the log prob of the corresponding hypothesis (from trans)
            log_probs: log-probs of the next word. Shape: (beam_size (len(trans)), |V|)
        """
        if isles != []:
            unfixed_isles = filter(lambda x: not is_sublist(x[1], fixed_words.values()), [isle for isle in isles])
        else:
            unfixed_isles = []
        trans = [[]]
        costs = [0.0]
        last_fixed_word = 0
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
            log_probs = numpy.log(self.comp_next_probs(c, k, last_words, *states)[0])  # c: representation, k: step_num, last_words: gen_y, *states: current_states
            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:, self.unk_id] = -numpy.inf
            if k < minlen:
                log_probs[:, self.eos_id] = -numpy.inf
            # If the current position is fixed, we fix it.
            if k in fixed_words: # This position is fixed by the user
                # We fix the word
                log_probs[:, :] =  -numpy.inf
                log_probs[:, fixed_words[k]] = 0.
                #last_fixed_position = fixed_positions.pop(0)

                # And continue with the regular process:
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
                new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                              in range(num_levels)]
                inputs = numpy.zeros(n_samples, dtype="int64")
                for i, (orig_idx, next_word, next_cost) in enumerate(
                        zip(trans_indices, word_indices, costs)):
                    new_trans[i] = trans[orig_idx] + [next_word]
                    new_costs[i] = next_cost
                    for level in range(num_levels):
                        new_states[level][i] = states[level][orig_idx]
                    inputs[i] = next_word
                new_states = self.comp_next_states(c, k, inputs, *new_states)

                # Filter the sequences that end with end-of-sequence character
                trans = []
                costs = []
                indices = []
                for i in range(n_samples):
                    if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                        trans.append(new_trans[i])
                        costs.append(new_costs[i])
                        indices.append(i)
                    else:
                        n_samples -= 1
                        if new_costs[i] != numpy.inf:
                            fin_trans.append(new_trans[i])
                            fin_costs.append(new_costs[i])
                states = map(lambda x: x[indices], new_states)

            else: # Position not fixed by the user
                if len(unfixed_isles) == 0: # There are no remaining isles. Regular decoding.
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
                    new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                                  in range(num_levels)]
                    inputs = numpy.zeros(n_samples, dtype="int64")
                    for i, (orig_idx, next_word, next_cost) in enumerate(
                            zip(trans_indices, word_indices, costs)):
                        new_trans[i] = trans[orig_idx] + [next_word]
                        new_costs[i] = next_cost
                        for level in range(num_levels):
                            new_states[level][i] = states[level][orig_idx]
                        inputs[i] = next_word
                    new_states = self.comp_next_states(c, k, inputs, *new_states)

                    # Filter the sequences that end with end-of-sequence character
                    trans = []
                    costs = []
                    indices = []
                    for i in range(n_samples):
                        if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                            trans.append(new_trans[i])
                            costs.append(new_costs[i])
                            indices.append(i)
                        else:
                            n_samples -= 1
                            if new_costs[i] != numpy.inf:
                                fin_trans.append(new_trans[i])
                                fin_costs.append(new_costs[i])
                    states = map(lambda x: x[indices], new_states)

                else: # We are in the middle of two isles
                    #logger.debug("Position %d in the middle of two isles"% k)
                    hyp_trans = [[]]*max_N
                    hyp_costs = [[]]*max_N
                    trans_ = copy.copy(trans)
                    costs_ = copy.copy(costs)
                    states_ = map(lambda x: [numpy.asarray(x, dtype="float32")], states)
                    states_ = states_*(max_N+1)
                    for kk in range(max_N):
                        beam_size = len(trans)
                        last_words = (numpy.array(map(lambda t: t[-1], trans_))
                                      if k+kk > 0
                                      else numpy.zeros(beam_size, dtype="int64"))
                        log_probs = numpy.log(self.comp_next_probs(c, k+kk, last_words, *states_[kk])[0])  # c: representation, k: step_num, last_words: gen_y, *states: current_states
                        # Adjust log probs according to search restrictions
                        if ignore_unk:
                            log_probs[:, self.unk_id] = -numpy.inf
                        if k+kk < minlen:
                            log_probs[:, self.eos_id] = -numpy.inf
                        if k+kk in fixed_words: # This position is fixed by the user
                            log_probs[:, :] =  -numpy.inf
                            log_probs[:, fixed_words[k+kk]] = 0.
                        # Find the best options by calling argpartition of flatten array
                        next_costs = numpy.array(costs_)[:, None] - log_probs
                        flat_next_costs = next_costs.flatten()
                        best_costs_indices = argpartition(
                            flat_next_costs.flatten(),
                            n_samples)[:n_samples]

                        # Decypher flatten indices
                        voc_size = log_probs.shape[1]
                        trans_indices = best_costs_indices / voc_size
                        word_indices = best_costs_indices % voc_size
                        costs_ = flat_next_costs[best_costs_indices]

                        # Form a beam for the next iteration
                        new_trans = [[]] * n_samples
                        new_costs = numpy.zeros(n_samples)
                        new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                                      in range(num_levels)]
                        inputs = numpy.zeros(n_samples, dtype="int64")
                        for i, (orig_idx, next_word, next_cost) in enumerate(
                                zip(trans_indices, word_indices, costs_)):
                            new_trans[i] = trans_[orig_idx] + [next_word]
                            new_costs[i] = next_cost
                            for level in range(num_levels):
                                new_states[level][i] = states_[kk][level][orig_idx]
                            inputs[i] = next_word
                        new_states = self.comp_next_states(c, k+kk, inputs, *new_states)

                        # Filter the sequences that end with end-of-sequence character
                        trans_ = []
                        costs_ = []
                        indices_ = []
                        for i in range(n_samples):
                            trans_.append(new_trans[i])
                            costs_.append(new_costs[i])
                            indices_.append(i)
                        states_[kk+1] = numpy.asarray(map(lambda x: x[indices_], new_states))
                        hyp_trans[kk] = trans_
                        hyp_costs[kk] = costs_

                    best_n_words = 0
                    min_cost = inf
                    best_hyp = []
                    for n_words in range(len(hyp_costs)):
                        for beam_index in range(len(hyp_costs[n_words])):
                            normalized_cost = hyp_costs[n_words][beam_index]/(n_words+1)
                            if normalized_cost < min_cost:
                                min_cost = normalized_cost
                                best_n_words = n_words
                                best_hyp = hyp_trans[n_words][beam_index]

                    trans = hyp_trans[best_n_words]
                    costs = hyp_costs[best_n_words]
                    states = states_[best_n_words+1]
                    #logger.debug("Generated %d words"%(best_n_words+1))
                    # We fix the words of the next isle
                    stop = False
                    isle_index = 0
                    while not stop and isle_index < len(unfixed_isles) and len(unfixed_isles) > 0:
                        k_counter = k + best_n_words + 1
                        next_isle = unfixed_isles[isle_index][1]
                        isle_prefixes = [next_isle[:i+1] for i in range(len(next_isle))]
                        #hyp = map (lambda x: idx2word[x] if idx2word.get(x) is not None else self.unk_id, best_hyp)
                        #logger.debug("Hypothesis:%s"%str([(hyp[i],i ) for i in range(len(hyp))]))
                        #logger.debug("Isle: %s"%str(map (lambda x: idx2word[x] if idx2word.get(x) is not None else self.unk_id, next_isle)))

                        _, start_pos = subfinder(next_isle, best_hyp)
                        if start_pos > -1:
                            k_counter = start_pos
                            #logger.debug("Isle included in hypothesis (position %d)"%(k_counter))
                            #logger.debug("Word %s (%d) will go to position %d"%(idx2word[next_isle[0]] if idx2word.get(next_isle[0]) is not None else self.unk_id, next_isle[0], k_counter))
                            del unfixed_isles[isle_index]

                        else:
                            for i in range(len(best_hyp)):
                                if any(map(lambda x: x == best_hyp[i:], isle_prefixes)):
                                    k_counter = i
                                    #logger.debug("Overlapping hypothesis and isle at position %d"%(k_counter))
                                    #logger.debug("Word %s (%d) will go to position %d"%(idx2word[next_isle[0]] if idx2word.get(next_isle[0]) is not None else self.unk_id, next_isle[0], k_counter))
                                    stop = True
                                    del unfixed_isles[isle_index]
                                    break
                        if k_counter ==  k + best_n_words + 1:
                            #logger.debug("Isle not included nor overlapped")
                            stop = True
                        for word in next_isle:
                            if fixed_words.get(k_counter) is None:
                                fixed_words[k_counter] = word
                            k_counter += 1
                        k += best_n_words
                    #logger.debug("Fixed_words: %s"%str(fixed_words))
            k+=1
        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, fixed_words=fixed_words,isles = isles,
                                   max_N=max_N, ignore_unk=False, minlen=minlen, verbose=verbose,
                                   idx2word=idx2word)
            elif n_samples < 500:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seq, n_samples * 2, fixed_words=fixed_words, isles = isles,
                                   max_N=max_N, ignore_unk=ignore_unk, minlen=minlen, verbose=verbose,
                                   idx2word=idx2word)
            else:
                logger.error("Translation failed")

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

def remove_from_list (list, symbol='</s>'):
    return filter(lambda a: a != symbol, list)

def is_sublist(list1, list2):
    return set(list2).issuperset(set(list1))

def subfinder(pattern, mylist):
    matches = []
    for start_pos in range(len(mylist)):
        if mylist[start_pos] == pattern[0] and mylist[start_pos:start_pos+len(pattern)] == pattern:
            return pattern, start_pos
    return [], -1

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
    p = (p-1)/p.shape[0]
    q = (q-1)/q.shape[0]

    return numpy.sum(p * numpy.log(p / q),0)

def sample(lm_model, seq, n_samples, fixed_words={}, max_N = -1, isles = [],
           sampler=None, beam_search=None, ignore_unk=False, normalize=False,
           alpha=1, verbose=False, idx2word=None):
    if beam_search:
        sentences = []
        if fixed_words is None or fixed_words == {}:
            minlen = len(seq) / 2
        else:
            minlen = max(len(seq) / 2, max(fixed_words.keys()))

        trans, costs = beam_search.search(seq, n_samples, fixed_words=fixed_words, isles = isles,
                                          max_N= max_N, ignore_unk=ignore_unk, minlen=minlen, verbose=verbose,
                                          idx2word=idx2word)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose:
                logger.log(2,"{}: {}".format(costs[i], sentences[i]))
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
            probs = cond_probs[:, sidx]
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


def parse_args():
    parser = argparse.ArgumentParser(
        "Sample (of find with beam-serch) translations from a translation model")
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
    parser.add_argument("--verbose",
                        action="store_true", default=False,
                        help="Be verbose")
    parser.add_argument("--max-n",
                        type=int, default=5, help="Maximum number of words generated between isles")
    parser.add_argument("model_path",
                        help="Path to the model")
    parser.add_argument("--interactive",
                        default=False, action="store_true",
                        help="Interactive post-editing?")
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


    if args.verbose:
        logger.setLevel(level=logging.DEBUG)
        logger.debug("I'm being verbose!")
    else:
        logger.setLevel(level=logging.INFO)

    logger.info('Post editing via the isles approach')

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)

    indx_word = cPickle.load(open(state['word_indx'], 'rb'))
    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_dec)
        beam_search.compile()
    else:
        sampler = enc_dec.create_sampler(many_samples=True)

    idict_src = cPickle.load(open(state['indx_word'], 'r'))
    unk_id = state['unk_sym_target']


    word2index = cPickle.load(open(state['word_indx_trgt'], 'r'))
    indx2word_trg = cPickle.load(open(state['indx_word_target'], 'rb'))
    word2index[""] = -1
    indx2word_trg[-1] = ""

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
            total_selection_effort = 0
            max_N = args.max_n
            if args.interactive:
                logger.info("Interactive mode\n\n")
                for n_line, line in enumerate(fsrc):
                    errors_sentence = 0
                    seqin = line.strip()
                    seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
                    hypothesis_number = 0
                    fixed_words_user = OrderedDict()
                    validated_hypothesis = False
                    stage = 0
                    isles = []
                    while not validated_hypothesis:
                        print ""
                        sentences, costs, _ = sample(lm_model, seq, n_samples, max_N=max_N, isles = isles,
                                                 fixed_words=fixed_words_user,
                                                 sampler=sampler, beam_search=beam_search, ignore_unk=args.ignore_unk,
                                                 normalize=args.normalize, verbose=args.verbose, idx2word=indx2word_trg)
                        best = numpy.argmin(costs)
                        hypothesis = sentences[best].split()
                        if args.color:
                            print_hypothesis = map(lambda x: 'colored(\''+x+'\', \'green\')' if fixed_words_user.get(hypothesis.index(x)) is not None else 'str(\''+x+'\')', hypothesis)
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
                                    fixed_positions = raw_input('Select the correct words (1 - %d). 0 means no correct words ' % len(hypothesis))
                                    fixed_positions = fixed_positions.split()
                                    fixed_positions = [int(word_pos)-1 for word_pos in fixed_positions]
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

                                    for i in range(0, correct_word-2):
                                        fixed_words_user[i] = word2index[hypothesis[i]]
                                    fixed_words_user[correct_word-1]    = word2index[new_word]
                                    break
                                else:
                                    print "Unknown action."
            else:
                #  Automatic mode
                for n_line, line in enumerate(fsrc):
                    errors_sentence = 0
                    hypothesis_number = 0

                    seqin = line.strip()
                    seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)

                    if args.verbose:
                        logger.debug("\n \n Processing sentence %d" % (n_line + 1))
                        logger.debug("Source: %s" % line[:-1])
                        logger.debug("Target: %s" % target_lines[n_line])

                        #print "\n \n Processing sentence %d" % (n_line + 1)
                        #print "Source: %s" % line[:-1]
                        #print "Desired translation: %s" % target_lines[n_line]

                    reference = target_lines[n_line].split()
                    # 0. Get a first hypothesis
                    sentences, costs, trans = sample(lm_model, seq, n_samples, sampler=sampler,
                                             beam_search=beam_search, ignore_unk=args.ignore_unk,
                                             normalize=args.normalize, verbose=args.verbose, idx2word=indx2word_trg)
                    hypothesis_number += 1
                    best = numpy.argmin(costs)
                    hypothesis = sentences[best].split()
                    show_hypothesis = copy.copy(hypothesis)
                    if args.save_original:
                        print >> ftrans_ori, " ".join(hypothesis)
                    if args.verbose:
                        logger.debug("Hypo_%d: %s"%(hypothesis_number, " ".join(hypothesis)))

                    if hypothesis == reference:
                        # If the sentence is correct, we  validate it
                        pass
                    else:
                        checked_index_r = 0
                        checked_index_h = 0
                        unk_words = []
                        unk_indices = []
                        n = 1
                        fixed_words_user = OrderedDict() # {pos: word}
                        while checked_index_r < len(reference):

                        #   2. Generate a new hypothesis considering this selection:
                        #       2.1. Selected words are fixed in their position
                        #       2.2. Non selected words are forbidden in that position
                        #   3. Test this new hypothesis
                            # Stage 1: Isles selection
                            #   1. Select the multiple isles in the hypothesis.
                            isles = find_isles(remove_from_list(hypothesis), reference)
                            hypothesis_isles = isles[0]
                            isle_indices = [(index, map (lambda x: word2index[x] if word2index.get(x) is not None
                                                                       else unk_id, word))
                                          for index, word in hypothesis_isles]
                            if args.verbose:
                                logger.debug("Isles: %s"%(str(hypothesis_isles)))

                            #TODO: Actualizar las islas en lugar de tener que seleccionarlas siempre desde cero
                            total_selection_effort += len(hypothesis_isles)
                            # Stage 2: Regular post editing
                            # From left to right, we will correct the hypotheses, taking into account the isles info
                            # At each timestep, the user can make three operations:
                            # Insertion of a new word at the end of the hypothesis
                            # Substitution of a word by another
                            # Deletion of the last part of a hypothesis

                            while checked_index_r < len(reference):  # We check all words in the reference
                                if checked_index_h >= len(hypothesis):
                                    # Insertions (at the end of the sentence)
                                    errors_sentence += 1
                                    new_word = reference[checked_index_r]
                                    fixed_words_user[checked_index_h] = word2index[new_word] \
                                        if word2index.get(new_word) is not None else unk_id
                                    if word2index.get(new_word) is None:
                                        unk_words.append(new_word)
                                        unk_indices.append(checked_index_h)
                                    if args.verbose:
                                        logger.debug('"%s" to position %d (end-of-sentence)'%(str(new_word), checked_index_h))

                                    #    logger.debug('\t Error case 0! ->'
                                    #                 'Add new word " % s" to the end of the hypothesis. '
                                    #                 % (new_word))
                                    break

                                elif hypothesis[checked_index_h] != reference[checked_index_r]:
                                    errors_sentence += 1
                                    new_word = reference[checked_index_r]
                                    # Substitution
                                    new_word_index = word2index[new_word] \
                                                if word2index.get(new_word) is not None else unk_id
                                    fixed_words_user[checked_index_h] = new_word_index
                                    #logger.debug('Adding word "%s" to position %d in fixed_words'%(str(new_word), checked_index_h))
                                    #logger.debug("%s"%str(fixed_words_user))
                                    if word2index.get(new_word) is None:
                                            if checked_index_h not in unk_indices:
                                                unk_words.append(new_word)
                                                unk_indices.append(checked_index_h)
                                    if args.verbose:
                                        logger.debug('"%s" to position %d'%(str(new_word), checked_index_h))


                                    #    logger.debug('\t Error case 2! ->Substitution of word "%s" '
                                    #                 '(index %d in the hypothesis) by word "%s" '
                                    #                 '(index %d in the reference). '
                                    #                 % (hypothesis[checked_index_h], checked_index_h,
                                    #                    new_word, checked_index_r))
                                    break
                                else:
                                    # No errors
                                    fixed_words_user[checked_index_h] = word2index[hypothesis[checked_index_h]] \
                                        if word2index.get(hypothesis[checked_index_h]) is not None else unk_id
                                    #if args.verbose:
                                    #    logger.debug('No error. Adding word "%s" to position %d in fixed_words'%(reference[checked_index_r], checked_index_h))
                                    #    logger.debug("%s"%str(fixed_words_user))
                                    checked_index_h += 1
                                    checked_index_r += 1
                            # We make compatible both approaches: Isles selection and prefixes
                            # Combination of the dictionaries:
                            #if args.verbose:
                            #    logger.debug("Isles_indices: %s"%(str(isle_indices)))
                            #    fix_w = []
                            #    for k in sorted(fixed_words_user.keys()):
                            #            fix_w.append('(' + str(k) + ', ' + indx2word_trg[fixed_words_user[k]] + ') ')
                            #    logger.debug("Fixed_words_user: %s" % (str(fix_w)))

                            # Generate a new hypothesis
                            logger.debug("")
                            sentences, costs, _ = sample(lm_model, seq, n_samples,
                                                 fixed_words=copy.copy(fixed_words_user), max_N = max_N, isles = isle_indices,
                                                 sampler=sampler, beam_search=beam_search, ignore_unk=args.ignore_unk,
                                                 normalize=args.normalize, verbose=args.verbose, idx2word=indx2word_trg)
                            hypothesis_number += 1

                            best = numpy.argmin(costs)
                            hypothesis = sentences[best].split()
                            if args.verbose:
                                logger.debug("Target: %s" % target_lines[n_line])
                                logger.debug("Hyp_%d: %s" % (hypothesis_number, " ".join(hypothesis)))
                            if len(unk_indices) > 0:  # If we added some UNK word
                                if len(hypothesis) < len(unk_indices):  # The full hypothesis will be made up UNK words:
                                    for i, index in enumerate(range(0, len(hypothesis))):
                                        hypothesis[index] = unk_words[unk_indices[i]]
                                    for ii in range(i+1, len(unk_words)):
                                        hypothesis.append(unk_words[ii])
                                else:  # We put each unknown word in the corresponding gap
                                    for i, index in enumerate(unk_indices):
                                        if index < len(hypothesis):
                                            hypothesis[index] = unk_words[i]
                                        else:
                                            hypothesis.append(unk_words[i])
                            if hypothesis == reference:
                                break
                        # Final check: The reference is a subset of the hypothesis: Cut the hypothesis
                        if len(reference) < len(hypothesis):
                            hypothesis = hypothesis[:len(reference)]
                            errors_sentence += 1
                            #logger.debug("Error case 3! -> Cut hypothesis. Errors: %d" % errors_sentence)

                    total_cost += costs[best]
                    total_errors += errors_sentence
                    total_words += len(hypothesis)
                    if args.verbose:
                        logger.debug("Final hypotesis: %s" % " ".join(hypothesis))
                        logger.debug("%d errors\n\n\n\n\n\n" % errors_sentence)

                    assert hypothesis == reference, "Error: The final hypothesis does not match with the reference! " \
                                                    "Sentence: %d \n" \
                                                    "Hypothesis: %s\n" \
                                                    "Reference:  %s" %(i, hypothesis, reference)
                    print >> ftrans, " ".join(hypothesis)

                    if (n_line + 1) % 50 == 0:
                        ftrans.flush()
                        if args.save_original:
                            ftrans_ori.flush()
                        logger.info("%d sentences processed" % (n_line+1))
                        logger.info("Current speed is {} per sentence".format((time.time() - start_time) / (n_line + 1)))
                        logger.info("Current WSR is: %f" % (float(total_errors) / total_words))
                        logger.info("Current Selection ratio is: %f" % (float(total_selection_effort) / n_line))

            print "Total number of errors:", total_errors
            print "Total number selections", total_selection_effort
            print "WSR: %f" % (float(total_errors) / total_words)
            print "Total cost of the translations: {}".format(total_cost)

            fsrc.close()
            ftrans.close()
            if args.save_original:
                ftrans_ori.close()
        except KeyboardInterrupt:
            print 'Interrupted!'
            print "Total number of corrections (up to now):", total_errors
            print "WSR: %f" % ((float(total_errors) / total_words))
            print "SR: %f" %((float(total_selection_effort) / n_line))

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
                seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
                print "Parsed Input:", parsed_in
            except Exception:
                print "Exception while parsing your input:"
                traceback.print_exc()
                continue

            sample(lm_model, seq, n_samples, sampler=sampler,
                   beam_search=beam_search,
                   ignore_unk=args.ignore_unk, normalize=args.normalize,
                   alpha=alpha, verbose=True)


if __name__ == "__main__":

    main()
