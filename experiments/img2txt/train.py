#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import re
import numpy
import time
import experiments

from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD import SGD as SGD
from groundhog.trainer.SGD_momentum import SGD as SGD_momentum
from groundhog.mainLoop import MainLoop

# import experiments.img2txt
from experiments.img2txt import\
        RNNEncoderDecoder, get_batch_iterator,\
        sample# , parse_input
from experiments.img2txt.sample import BeamSearch


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
                # x_words = cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], x))
                y_words = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], y))
                #if len(x_words) == 0:
                #    continue

                # self.__print_samples("Input", x_words, self.state['source_encoding'])
                self.__print_samples("Target", y_words, self.state['target_encoding'])

                #if self.state['source_encoding'] == "uft8":
                #    print u"Input: {}".format(" ".join(x_words))
                #elif self.state['source_encoding'] == "ascii":
                #    print "Input: {}".format(" ".join(x_words))
                #if self.state['target_encoding'] == "utf8":
                #    print u"Target: {}".format(" ".join(y_words))
                #elif self.state['target_encoding'] == "ascii":
                #    print "Target: {}".format(" ".join(y_words))

		self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], x[:len(y_words)])
                sample_idx += 1
    def __print_samples(self, output_name, words, encoding):
        if encoding == "utf8":
            # print u"{}: {}".format(output_name, " ".join(words))
            print output_name;": ", " ".join(words).decode('utf-8')

        elif encoding == "ascii":
            print "{}: {}".format(output_name, " ".join(words))
        else:
            print "Unknown encoding {}".format(encoding)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_img_search_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    parser.add_argument("--skip-train", action="store_true", default=False,
        help="Skip training (only loads data and compile trainer, for debugging purposes)")
    return parser.parse_args()

def main():
    args = parse_args()

    # this loads the state specified in the prototype

    state = getattr(experiments.img2txt, args.proto)()
    # this is based on the suggestion in the README.md in this foloder
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
    if args.skip_train == True :
        logger.warning("Skipping training!")
    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, args.skip_init)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()


    # If we are going to use validation with the bleu script, we
    # will need early stopping
    bleu_validator = None
    if state['bleu_script'] is not None and state['validation_set'] is not None\
        and state['validation_set_grndtruth'] is not None:
        # make beam search
        beam_search = BeamSearch(enc_dec)
        beam_search.compile()
        # bleu_validator = BleuValidator(state, lm_model, beam_search, verbose=state['output_validation_set'])
    else :
        logger.warning("BLEU validation will not be used")
    logger.debug("Load data")
    train_data = get_batch_iterator(state)
    logger.debug("Compile trainer")

    algo = eval(state['algo'])(lm_model, state, train_data)
    if args.skip_train == True :
        logger.warning("Skipping training!")
    else :
        logger.debug("Running training")
    main = MainLoop(train_data, None, None, lm_model, algo, state, None,
            reset=state['reset'],
            bleu_val_fn = bleu_validator,
            # skip_train = args.skip_train,
            hooks=[RandomSamplePrinter(state, lm_model, train_data)]
                if state['hookFreq'] >= 0 #and state['validation_set'] is not None
                else None)

    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        logger.debug("Run!!")
        main.main()

if __name__ == "__main__":
    main()
