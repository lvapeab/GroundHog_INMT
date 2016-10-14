#!/usr/bin/env python

import os, sys
import cPickle
import argparse
import numpy
import logging

from groundhog.datasets.UnbufferedDataIterator import UnbufferedDataIterator
from online_trainer import Trainer

from groundhog.trainer.SGD_online import SGD
from groundhog.trainer.SGD_adagrad import AdaGrad

from experiments.nmt import prototype_state, prototype_search_state, \
                            RNNEncoderDecoder

from experiments.nmt.sample import BeamSearch, sample
from experiments.nmt.language import loadSourceLanguageFromState, \
                                     loadTargetLanguageFromState

logging.getLogger("experiments.nmt.sample").addHandler(logging.StreamHandler())

class Sampler:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def translate(self, seq):
        hypos, cost, _ = sample(seq=seq, **self.__dict__)
        if len(cost) == 0:
            logging.debug('No available hypotheses.')
            return None
        else:
            return hypos[numpy.argmin(cost)]

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_language", help="Source language")
    parser.add_argument("target_language", help="Target language")
    parser.add_argument("source_file", help="Source file")
    parser.add_argument("target_file", help="Target file")
    parser.add_argument("state", help="State to use")

    parser.add_argument("-v", action="store_true", help="Verbose mode")
    parser.add_argument("-f", action="store_true", default=True, help="Rename output path in case it exists")
    parser.add_argument("-o", action="store_true", help="Redirect text output to file in generated output prefix")
    parser.add_argument("-n", action="store_true", help="Add noise to weights and inputs")
    parser.add_argument("-d", action="store_true", help="Use dropout")
    parser.add_argument("-a", "--always-compute-hypotheses",
                        action="store_true",
                        help="Compute hypotheses for every training epoch")
    parser.add_argument("-m", "--model", help="Path to the trained model file. If absent, one is created with the state setup.", default=None)
    parser.add_argument("-t", "--tokenizer", help="Path to tokenizer script")
    parser.add_argument("-s", "--save-frequency", help="Every how many epochs save the model. -1 to never save it until the end.", type=int, default=-1)
    parser.add_argument("-e", "--num-epochs", help="Train for how many epochs. Defaults to 1.", type=int, default=1)
    parser.add_argument("-l", "--learning-rate", help="Initial learning rate.")
    parser.add_argument("-z", "--sequence-length", help="Sequence length", type=int, default=None)
    parser.add_argument("-i", "--multiple-iterations", help="Perform multiple training iterations for each sample", type=int, default=1)
    parser.add_argument("-g", "--algorithm", help="Algorithm to use for training. Defaults to AdaGrad.", default='AdaGrad')
    parser.add_argument("--tolerance", help="Tolerance parameter of PA. Only valid if --algorithm is PassiveAggressive.", default=None)
    parser.add_argument("--max-samples", help="Use only a number of samples from the beginning. Useful for experiments.", type=int, default=None)
    parser.add_argument("--iterate-until-convergence", help="For " + \
                        "Passive-Aggressive: Iterate with a single sample until the weight " + \
                        "updates are very small.", action="store_true")

    return parser.parse_args()

def readState(path):
    state = prototype_search_state()
    if path.endswith(".py"):
        state.update(eval(open(path).read()))
    else:
        with open(path) as src:
            state.update(cPickle.load(src))
    return state

def getModel(state, create_sampler, model_path = None):
    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    beam_search = None
    if create_sampler:
        beam_search = BeamSearch(enc_dec)
        beam_search.compile()
    lm_model = enc_dec.create_lm_model()
    if model_path is not None:
        lm_model.load(model_path)
    else:
        logging.debug('EncDec model created from state setup')
    return lm_model, beam_search, enc_dec.predictions.word_probs

def loadAlgorithm(state, model, batch_iter, algorithm_name,
                  word_probs = None, seqlen = None, iterations = None,
                  tolerance = None, iterate_until_convergence = None):
    logging.debug('Loading algorithm...')
    if seqlen is None:
        seqlen = state['seqlen']
    if algorithm_name == 'PassiveAggressive':
        # PA accepts extra parameters
        algo = eval(algorithm_name)(model, state, batch_iter,
                                    word_probs, seqlen, iterations,
                                    tolerance, iterate_until_convergence)
    else:
        algo = eval(algorithm_name)(model, state, batch_iter)
    return algo

def resolve_output_path_conflicts(directory, force):
    if os.path.exists(directory):
        if not force:
            raise Exception(('The path %s points to an existing directory or file. ' \
                             + 'Please, rename it or delete it before proceeding, ' \
                             + 'or use the flag -f.') % directory)
        else:
            try:
                os.rename(directory, directory + '.old')
            except:
                raise Exception(('Both %s and %s.old point to existing directories or files. ' \
                                 + 'The conflict could not be resolved automatically. ' \
                                 + 'Please, free the path %s before proceeding.') % (directory,
                                                                                     directory,
                                                                                     directory))
            logging.debug('The file or directory at %s has been moved to %s.old.' %
                          (directory, directory))

def getOutputPrefix(statePath, source_file, algorithm_name,
                    lr, noise, dropout, model, numIterations,
                    tolerance, max_samples):
    if algorithm_name == 'PassiveAggressive':
        algorithm_name += "%.9f" % tolerance

    state_directory = os.path.dirname(os.path.abspath(statePath))
    source_file_without_extension = os.path.basename(source_file)
    if '.' in source_file:
        source_file_without_extension = source_file_without_extension[:source_file_without_extension.rindex('.')]
    output_prefix = '%s/%s_%s%s_%f%s%s%s%s/' % (state_directory, algorithm_name,
                                                source_file_without_extension,
                                                ("%d" % max_samples)
                                                if max_samples is not None
                                                else "",
                                                lr, '_noise' if noise else '',
                                                '_dropout' if dropout != 1 else '',
                                                '_scratch' if model is None else '',
                                                '_%diterations' % numIterations if numIterations > 1 else '')
    return output_prefix

def setupLogging(outputFile, level):
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logging.basicConfig(filename=outputFile,
                        filemode='w',
                        format='%(levelname)s:%(filename)s: %(message)s',
                        level=level)

    class FakeFile:
        def __init__(self):
            self.currentMessage = ''
        def write(self, message):
            nextln = message.find('\n')
            if nextln == -1:
                self.currentMessage += message + ' '
            else:
                while nextln != -1:
                    self.currentMessage += message[:nextln]
                    logging.debug(self.currentMessage)
                    self.currentMessage = ''
                    message = message[nextln + 1:] # So as to skip line break
                    nextln = message.find('\n')

    sys.stdout = FakeFile()
    sys.stderr = sys.stdout

def main():
    args = parseArguments()

    tolerance = args.tolerance
    if tolerance is not None:
        tolerance = float(tolerance)

    assert not (args.algorithm == 'PassiveAggressive' and tolerance is None), \
    'tolerance parameter must be used when algorithm is PassiveAggressive'

    max_samples = args.max_samples
    if max_samples is not None:
        max_samples = int(max_samples)

    verbose = args.v
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    state = readState(args.state)

    state['weight_noise'] = args.n
    state['dropout'] = 0.5 if args.d else 1
    # dropout_rec must be 1 (off) always

    if args.learning_rate is not None:
        state['lr'] = float(args.learning_rate)


    # Next [:-1] is there so the last slash is not included (could be
    # a file if it already exists)
    output_prefix = getOutputPrefix(args.state, args.source_file, args.algorithm,
                                    state['lr'], state['weight_noise'], state['dropout'],
                                    args.model, int(args.multiple_iterations),
                                    tolerance, max_samples)
    forceOutputPath = args.f
    resolve_output_path_conflicts(output_prefix[:-1], forceOutputPath)
    os.makedirs(output_prefix)

    print 'Output files can be found at: %s' % output_prefix

    outputFile = None
    if args.o:
        outputFile = output_prefix + 'output'

    setupLogging(outputFile, level)

    if args.algorithm == 'PassiveAggressive' and args.learning_rate is not None:
        logging.warning('Learning rate is not used in PA, remove this CLI parameter to remove this warning')

    if args.algorithm != 'PassiveAggressive' and tolerance is not None:
        logging.warning('Tolerance is only used in PA, remove this CLI parameter to remove this warning')

    if args.algorithm != 'PassiveAggressive' and args.iterate_until_convergence:
        logging.warning('Iteration until convergence is only used in PA, remove this CLI parameter to remove this warning')

    launchTrainer(state, args.source_file, args.target_file,
                  args.source_language, args.target_language,
                  args.model, args.sequence_length,
                  args.save_frequency, args.num_epochs,
                  args.always_compute_hypotheses,
                  int(args.multiple_iterations), args.algorithm,
                  output_prefix, tolerance, max_samples,
                  args.iterate_until_convergence, args.always_compute_hypotheses)

def logTrainingCPUTime(time, output_file):
    with open(output_file, 'w') as fileHandler:
        fileHandler.write(str(time))

def launchTrainer(state, sourceFile, targetFile,
                  sourceLanguage, targetLanguage,
                  model, sequenceLength,
                  saveFrequency, numEpochs,
                  alwaysComputeHypotheses,
                  numIterations, algorithm_name,
                  output_prefix, tolerance,
                  max_samples, iterate_until_convergence,
                  always_compute_hypotheses):

    assert numIterations == 1 or numEpochs == 1

    # Parameters
    ignore_unk = False
    normalize = False
    beam_size = 20

    source_file = sourceFile
    target_file = targetFile
    num_sentences = 1

    sourceLanguage = loadSourceLanguageFromState(state, sourceLanguage)
    targetLanguage = loadTargetLanguageFromState(state, targetLanguage)

    computing_hypotheses = model is not None or always_compute_hypotheses

    if sequenceLength is not None:
        state['seqlen'] = int(sequenceLength)

    # Those are to check in the output that languages are correctly defined
    logging.debug('Source file: %s' % source_file)
    logging.debug('Target file: %s' % target_file)
    logging.debug('Source dictionary path: %s' % state['indx_word'])
    logging.debug('Target dictionary path: %s' % state['indx_word_target'])
    logging.debug('Number of sentences: %d' % num_sentences)
    logging.debug('Sequence length: %d' % state['seqlen'])

    batch_iter = UnbufferedDataIterator(source_file, target_file, state,
                                        sourceLanguage.word_indx, targetLanguage.word_indx,
                                        sourceLanguage.indx_word, targetLanguage.indx_word,
                                        num_sentences, state['seqlen'],
                                        max_samples)

    logging.debug("Fetching model...")
    model, beam_search, word_probs = getModel(state, computing_hypotheses, model)
    print "state=", state
    if algorithm_name == 'PassiveAggressive':
        algo = loadAlgorithm(state, model, batch_iter, algorithm_name, word_probs,
                             state['seqlen'], numIterations, tolerance, iterate_until_convergence)
    else:
        algo = loadAlgorithm(state, model, batch_iter, algorithm_name)

    sampler = Sampler(lm_model=model, n_samples=beam_size, sampler=None, beam_search=beam_search,
                      ignore_unk=ignore_unk, normalize=normalize)

    iteration = 0
    save_model_frequency = int(saveFrequency)
    maxEpochs = int(numEpochs)

    if numIterations == 1:
        logging.debug('Will train for %d epochs.' % maxEpochs)
    else:
        logging.debug('Will train %d iterations per sample.' % numIterations)

    savingModelPolicy = 'last_epoch'
    if save_model_frequency > 0:
        logging.debug('Saving model every %d iterations' % save_model_frequency)
        savingModelPolicy = save_model_frequency

    hypothesesPolicy = 'never'
    if alwaysComputeHypotheses:
        hypothesesPolicy = 'always'
    elif computing_hypotheses:
        hypothesesPolicy = 'last_epoch'

    if numIterations <= 1 or algorithm_name == 'PassiveAggressive':
        trainer = Trainer(model, batch_iter, algo, maxEpochs, output_prefix,
                          hypothesesPolicy, sampler, savingModelPolicy)

    trainer.train()

if __name__ == "__main__":
    main()
