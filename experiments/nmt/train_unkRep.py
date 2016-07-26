#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import re
import numpy
import time
import experiments
import theano
from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD import SGD as SGD
from groundhog.trainer.SGD_momentum import SGD as SGD_momentum



from groundhog.mainLoop import MainLoop
from experiments.nmt import\
        RNNEncoderDecoder, prototype_state, get_batch_iterator,\
        sample, parse_input
import experiments.nmt
from experiments.nmt.sample import BeamSearch
from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

class RandomSamplePrinter(object):

    def __init__(self, state, model, train_iter):
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]
            raise Exception("No end-of-line found")

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next(peek=True)
            xs, ys = batch['x'], batch['y']
            for seq_idx in range(xs.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break

                x, y = xs[:, seq_idx], ys[:, seq_idx]
                x_words = cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], x))
                y_words = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], y))
                if len(x_words) == 0:
                    continue

                self.__print_samples("Input", x_words, self.state['source_encoding'])
                self.__print_samples("Target", y_words, self.state['target_encoding'])

		self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], x[:len(x_words)])
                sample_idx += 1
    def __print_samples(self, output_name, words, encoding):
        if encoding == "utf8":
            # print u"{}: {}".format(output_name, " ".join(words))
            print output_name;": ", " ".join(words).decode('utf-8')

        elif encoding == "ascii":
            print "{}: {}".format(output_name, " ".join(words))
        else:
            print "Unknown encoding {}".format(encoding)

class BleuValidator(object):
    """
    Object that evaluates the bleu score on the validation set.
    Opens the subprocess to run the validation script, and
    keeps track of the bleu scores over time
    """
    def __init__(self, state, lm_model,
                beam_search, alignment_fns, heuristic, mapping, ignore_unk=False,
                normalize=False, verbose=False):
        """
        Handles normal book-keeping of state variables,
        but also reloads the bleu scores if they exists

        :param state:
            a state in the usual groundhog sense
        :param lm_model:
            a groundhog language model
        :param beam_search:
            beamsearch object used for sampling
        :param ignore_unk
            whether or not to ignore unknown characters
        :param normalize
            whether or not to normalize the score by the length
            of the sentence
        :param verbose
            whether or not to also write the ranslation to the file
            specified by validation_set_out

        """

        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

        self.indx_word = cPickle.load(open(state['word_indx'],'rb'))
        self.idict_src = cPickle.load(open(state['indx_word'],'r'))
        self.n_samples = state['beam_size']
        self.best_bleu = 0
        self.alignment_fns = alignment_fns
        self.heuristic = heuristic
        self.mapping = mapping,
        self.unk_id = state['unk_sym_target']

        self.val_bleu_curve = []
        if state['reload']:
            try:
                bleu_score = numpy.load(state['prefix'] + 'val_bleu_scores.npz')
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()
                print "BleuScores Reloaded"
            except:
                print "BleuScores not Found"

        if state['char_based_bleu']:
            self.multibleu_cmd = ['perl', state['bleu_script'], '-char', state['validation_set_grndtruth'], '<']
        else:
            self.multibleu_cmd = ['perl', state['bleu_script'], state['validation_set_grndtruth'], '<']

        if verbose:
            if 'validation_set_out' not in state.keys():
                self.state['validation_set_out'] = state['prefix'] + 'validation_out.txt'

    def __call__(self):
        """
        Opens the file for the validation set and creates a subprocess
        for the multi-bleu script.

        Returns a boolean indicating whether the current model should
        be saved.
        """

        print "Started Validation: "
        val_start_time = time.time()
        fsrc = open(self.state['validation_set'], 'r')
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        if self.verbose:
            ftrans = open(self.state['validation_set_out'], 'w')

        for i, line in enumerate(fsrc):
            """
            Load the sentence, retrieve the sample, write to file
            """
            if self.state['source_encoding'] == 'utf8':
                seqin = line.strip().decode('utf-8')
            else:
                seqin = line.strip()
            seq, parsed_in = parse_input(self.state, self.indx_word, seqin, idx2word=self.idict_src)
            src_words = parsed_in.split()

            # draw sample, checking to ensure we don't get an empty string back
            sentences, costs, trans = sample.sample(self.lm_model, seq, self.n_samples,
                    beam_search=self.beam_search, ignore_unk=self.ignore_unk, normalize=self.normalize)
            try:
                best = numpy.argmin(costs)
                total_cost += costs[best]
                hypothesis = sentences[best].split()
                trans_out = sentences[best]
                trg_seq = trans[best]
            except ValueError:
                print "Could not fine a translation for line: {}".format(i+1)
                trans_out = u'UNK' if self.state['target_encoding'] == 'utf8' else 'UNK'
            if self.state['unkReplace'] and self.state['oov'] in hypothesis:
                logger.debug("Unknown word in line %i"%i)
                hard_alignments = self.compute_alignment(seq, trg_seq, self.alignment_fns)
                hypothesis = self.replace_unknown_words(src_words, trg_seq, hypothesis, hard_alignments, self.unk_id,
                                heuristic=self.heuristic, mapping = self.mapping)

            # Write to subprocess and file if it exists
            if self.state['target_encoding'] == 'utf8' and \
                self.state['char_based_bleu']:
                print >> mb_subprocess.stdin, trans_out.encode('utf8').replace(" ","")
                if self.verbose:
                    print  >> ftrans, trans_out.encode('utf8').replace(" ","")
            elif self.state['target_words_segmented']:
                print >> mb_subprocess.stdin, \
                        self.append_suffixes(trans_out)#.encode(self.state['target_encoding']))
                if self.verbose:
                    print >> ftrans, \
                            self.append_suffixes(trans_out)#.encode(self.state['target_encoding']))
            else:
                print >> mb_subprocess.stdin, trans_out
                if self.verbose:
                    print >> ftrans, trans_out

            if i != 0 and i % 200 == 0:
                print "Translated {} lines of validation set...".format(i)
            mb_subprocess.stdin.flush()

        print "Total cost of the validation: {}".format(total_cost)
        fsrc.close()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        out_parse = re.match(r'BLEU = [-.0-9]+', mb_subprocess.stdout.readline())
        print "Validation Took: {} minutes".format(float(time.time() - val_start_time)/60.)
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        print bleu_score
        mb_subprocess.terminate()

        # Determine whether or not we should save
        if self.best_bleu < bleu_score:
            self.best_bleu = bleu_score
            print "Best_BLEU: %f"%self.best_bleu
            return True
        return False

    def compute_alignment(self, src_seq, trg_seq, alignment_fns):

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


    def replace_unknown_words(self, src_word_seq, trg_seq, trg_word_seq, hard_alignment, unk_id, heuristic=0, mapping=[]):
        trans_words = trg_word_seq
        trans_seq = trg_seq
        hard_alignment = hard_alignment
        new_trans_words = []
        for j in xrange(len(trans_words)): # -1 : Don't write <eos>
            if trans_seq[j] == unk_id:
                UNK_src = src_word_seq[hard_alignment[j]]
                if heuristic == 0:  # Copy (ok when training with large vocabularies on en->fr, en->de)
                    new_trans_words.append(UNK_src)
                    logger.debug('Copying word %s' %UNK_src)
                elif heuristic == 1:
                    # Use the most likely translation (with t-table). If not found, copy the source word.
                    # Ok for small vocabulary (~30k) models
                    if UNK_src in mapping:
                        new_trans_words.append(mapping[UNK_src])
                        logger.debug('%s found in mapping: %s.'%(UNK_src, mapping[UNK_src]))
                    else:
                        new_trans_words.append(UNK_src)
                        logger.debug('%s not found in mapping. Copying word.'%UNK_src)

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


    def append_suffixes(self,trans):
        '''
        Suffix merger for segmented words, looks for suffixes in <tag:_sfx_>
        pattern and appends _sfx_ to its corresponding stem

        :param trans:
            sentence to be merged, string

        '''
        out = []
        for word in trans.split():
            if word.startswith('<') and word.endswith('>'):
                sfx = re.search(r':(.*)\>',word)
                if sfx is not None:
                    out.append(sfx.group(1))
            else: # it is a stem
                if not out:
                    out.append(' ')
                out.append(word)
        return ''.join(out)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_search_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()

    # this loads the state specified in the prototype
    state = getattr(experiments.nmt, args.proto)()
    # this is based on the suggestion in the README.md.md in this foloder
    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("State:\n{}".format(pprint.pformat(state)))

    rng = numpy.random.RandomState(state['seed'])

    enc_dec = RNNEncoderDecoder(state, rng, args.skip_init, compute_alignment=state['unkReplace'])
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()

    if state['unkReplace']:
        alignment_fns = [theano.function(inputs=enc_dec.inputs,
                                                 outputs=[enc_dec.alignment], name="alignment_fn")]
        if state['mapping'] is not None:
            with open(state['mapping'], 'rb') as f:
                mapping = cPickle.load(f)
                logger.debug("Loaded mapping file from %s" % str(state['mapping']))
            heuristic = state['heuristic']
        else:
            heuristic = 0
            mapping = None
    logger.info("Replacing unkown words according to heuristic %d" % heuristic)


    # If we are going to use validation with the bleu script, we
    # will need early stopping
    bleu_validator = None
    if state['bleu_script'] is not None and state['validation_set'] is not None\
        and state['validation_set_grndtruth'] is not None:
        # make beam search
        beam_search = BeamSearch(enc_dec)
        beam_search.compile()
        bleu_validator = BleuValidator(state, lm_model, beam_search, alignment_fns, heuristic, mapping, verbose=state['output_validation_set'])
    else :
        logger.warning("BLEU validation will not be used")
    logger.debug("Load data")
    train_data = get_batch_iterator(state)
    logger.debug("Compile trainer")

    algo = eval(state['algo'])(lm_model, state, train_data)
    logger.debug("Run training")

    main = MainLoop(train_data, None, None, lm_model, algo, state, None,
            reset=state['reset'],
            bleu_val_fn = bleu_validator,
            hooks=[RandomSamplePrinter(state, lm_model, train_data)]
                if state['hookFreq'] >= 0 #and state['validation_set'] is not None
                else None)

    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        main.main()

if __name__ == "__main__":
    main()
