
import numpy
import cPickle
import logging

logger = logging.getLogger(__name__)

def loadFile(path):
    return cPickle.load(open(path, 'r'))

class Language:
    def __init__(self, eos_index, unk_index, oov_sym,
                 n_sym, lang, word_indx_path, indx_word_path):
        self.lang = lang
        self.eos_index = eos_index
        self.unk_index = unk_index
        self.oov_sym = oov_sym
        self.n_sym = n_sym

        self.word_indx = loadFile(word_indx_path)
        self.indx_word = loadFile(indx_word_path)
        self.indx_word[eos_index] = '<eol>'
        self.indx_word[unk_index] = oov_sym
        logger.debug('Language created: %s' % lang)
        logger.debug('First word in dict: %s' % self.word_indx.keys()[0])

    def indicesToWords(self, seq):
        try:
            text = ' '.join([self.indx_word[i] for i in seq])
        except LookupError as exception:
            raise LookupError("Index out of dictionary of language %s: %d" % (self.lang, exception.args[0]))
        return text

    def wordsToIndices(self, seq):
        # For compatibility with legacy function parse_input
        # this function adds an end-of-line word.
        words = seq.split()
        result = numpy.zeros(len(words) + 1, dtype='int64')

        for index, word in enumerate(words):
            result[index] = self.word_indx.get(words[index], self.unk_index)
        result[-1] = self.eos_index

        string = self.indicesToWords(result)
        return result, string


def loadSourceLanguageFromState(state, lang):
    return Language(state['null_sym_source'], state['unk_sym_source'],
                    state['oov'], state['n_sym_source'], lang,
                    state['word_indx'], state['indx_word'])

def loadTargetLanguageFromState(state, lang):
    return Language(state['null_sym_target'], state['unk_sym_target'],
                    state['oov'], state['n_sym_target'], lang,
                    state['word_indx_trgt'], state['indx_word_target'])

def create_padded_batch(x, y, seqlen, trim_batches, state, return_dict=False):
    """A callback given to the iterator to transform data in suitable format

    :type x: list
    :param x: list of numpy.array's, each array is a batch of phrases
        in some of source languages

    :type y: list
    :param y: same as x but for target languages

    :param new_format: a wrapper to be applied on top of returned value

    :returns: a tuple (X, Xmask, Y, Ymask) where
        - X is a matrix, each column contains a source sequence
        - Xmask is 0-1 matrix, each column marks the sequence positions in X
        - Y and Ymask are matrices of the same format for target sequences
        OR new_format applied to the tuple

    Notes:
    * actually works only with x[0] and y[0]
    * len(x[0]) thus is just the minibatch size
    * len(x[0][idx]) is the size of sequence idx
    """
    x = x[0]
    y = y[0]
    mx = seqlen
    my = seqlen
    if trim_batches:
        # Similar length for all source sequences
        mx = numpy.minimum(seqlen, max([len(xx) for xx in x]))+1
        # Similar length for all target sequences
        my = numpy.minimum(seqlen, max([len(xx) for xx in y]))+1
    # Batch size
    n = x.shape[0]
    if n == 1:
        x = x[0]
        y = y[0]
        X = numpy.zeros((len(x), 1), dtype = 'int64')
        X[:,0] = x
        Y = numpy.zeros((len(y), 1), dtype = 'int64')
        Y[:,0] = y
        Xmask = numpy.ones((len(x), 1), dtype = 'float32')
        Ymask = numpy.ones((len(y), 1), dtype = 'float32')
    else:
        X = numpy.zeros((mx, n), dtype='int64')
        Y = numpy.zeros((my, n), dtype='int64')
        Xmask = numpy.zeros((mx, n), dtype='float32')
        Ymask = numpy.zeros((my, n), dtype='float32')
        # Fill X and Xmask
        for sentenceIndex in xrange(len(x)):
            # Insert sequence sentenceIndex in a column of matrix X
            sentenceLength = len(x[sentenceIndex])
            # This is mainly done for the mask not to include two EOS
            if x[sentenceIndex][sentenceLength - 1] == state['null_sym_source']:
                sentenceLength -= 1
            realLength = min(mx, sentenceLength)
            X[:realLength, sentenceIndex] = x[sentenceIndex][:realLength]
            # Mark the end of phrase
            if sentenceLength < mx:
                X[sentenceLength:, sentenceIndex] = state['null_sym_source']
            # Initialize Xmask column with ones in all positions that
            # were just set in X
            Xmask[:sentenceLength, sentenceIndex] = 1.
            if sentenceLength < mx:
                Xmask[sentenceLength, sentenceIndex] = 1.
        # Fill Y and Ymask in the same way as X and Xmask in the previous loop
        for sentenceIndex in xrange(len(y)):
            sentenceLength = len(y[sentenceIndex])
            # This is mainly done for the mask not to include two EOS
            if y[sentenceIndex][sentenceLength - 1] ==  state['null_sym_target']:
                sentenceLength -= 1
            realLength = min(my, sentenceLength)
            Y[:realLength, sentenceIndex] = y[sentenceIndex][:realLength]
            if sentenceLength < my:
                Y[sentenceLength:, sentenceIndex] =  state['null_sym_target']
            Ymask[:sentenceLength, sentenceIndex] = 1.
            if sentenceLength < my:
                Ymask[sentenceLength, sentenceIndex] = 1.
        null_inputs = numpy.zeros(X.shape[1])
        # We say that an input pair is valid if both:
        # - either source sequence or target sequence is non-empty
        # - source sequence and target sequence have null_sym ending
        # Why did not we filter them earlier?
        for sentenceIndex in xrange(X.shape[1]):
            if numpy.sum(Xmask[:,sentenceIndex]) == 0 and numpy.sum(Ymask[:,sentenceIndex]) == 0:
                null_inputs[sentenceIndex] = 1
            if Xmask[-1,sentenceIndex] and X[-1,sentenceIndex] != state['null_sym_source']:
                null_inputs[sentenceIndex] = 1
            if Ymask[-1,sentenceIndex] and Y[-1,sentenceIndex] !=  state['null_sym_target']:
                null_inputs[sentenceIndex] = 1
        valid_inputs = 1. - null_inputs
        # Leave only valid inputs
        X = X[:,valid_inputs.nonzero()[0]]
        Y = Y[:,valid_inputs.nonzero()[0]]
        Xmask = Xmask[:,valid_inputs.nonzero()[0]]
        Ymask = Ymask[:,valid_inputs.nonzero()[0]]
        if len(valid_inputs.nonzero()[0]) <= 0:
            return None
        # Unknown words
        X[X >= state['n_sym_source']] = state['unk_sym_source']
        Y[Y >= state['n_sym_target']] = state['unk_sym_target']
    # -- end else
    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask
