import numpy as np
import logging
import cPickle


def create_batch_from_seqs(src, trg):

    return {'x': np.asarray([[src_word] for src_word in src]),
            'x_mask': np.asarray([[1.] for _ in src]),
            'y': np.asarray([[trg_word] for trg_word in trg]),
            'y_mask': np.asarray([[1.] for _ in trg])}



def loadFile(path):
    return cPickle.load(open(path, 'r'))




class Language:
    def __init__(self, eos_index, unk_index, oov_sym,
                 n_sym, word_indx_path, indx_word_path):
        self.eos_index = eos_index
        self.unk_index = unk_index
        self.oov_sym = oov_sym
        self.n_sym = n_sym

        self.word_indx = loadFile(word_indx_path)
        self.indx_word = loadFile(indx_word_path)
        self.indx_word[eos_index] = '<eol>'
        self.indx_word[unk_index] = oov_sym

    def indicesToWords(self, seq):
        try:
            text = ' '.join([self.indx_word[i] for i in seq])
        except LookupError as exception:
            raise LookupError("Index out of dictionary of language: %d" % (exception.args[0]))
        return text

    def wordsToIndices(self, seq):
        # For compatibility with legacy function parse_input
        # this function adds an end-of-line word.
        words = seq.split()
        result = np.zeros(len(words) + 1, dtype='int64')

        for index, word in enumerate(words):
            result[index] = self.word_indx.get(words[index], self.unk_index)
        result[-1] = self.eos_index

        string = self.indicesToWords(result)
        return result, string


def loadSourceLanguageFromState(state):
    return Language(state['null_sym_source'], state['unk_sym_source'],
                    state['oov'], state['n_sym_source'],
                    state['word_indx'], state['indx_word'])

def loadTargetLanguageFromState(state):
    return Language(state['null_sym_target'], state['unk_sym_target'],
                    state['oov'], state['n_sym_target'],
                    state['word_indx_trgt'], state['indx_word_target'])

def create_padded_batch(x, y, seqlen, trim_batches, source_language,
                        target_language, return_dict=False):
    """A callback given to the iterator to transform data in suitable format

    :type x: list
    :param x: list of np.array's, each array is a batch of phrases
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
        mx = np.minimum(seqlen, max([len(xx) for xx in x]))+1
        # Similar length for all target sequences
        my = np.minimum(seqlen, max([len(xx) for xx in y]))+1

    # Batch size
    n = x.shape[0]

    if n == 1:
        x = x[0]
        y = y[0]
        X = np.zeros((len(x), 1), dtype = 'int64')
        X[:,0] = x
        Y = np.zeros((len(y), 1), dtype = 'int64')
        Y[:,0] = y

        Xmask = np.ones((len(x), 1), dtype = 'float32')
        Ymask = np.ones((len(y), 1), dtype = 'float32')
    else:
        X = np.zeros((mx, n), dtype='int64')
        Y = np.zeros((my, n), dtype='int64')
        Xmask = np.zeros((mx, n), dtype='float32')
        Ymask = np.zeros((my, n), dtype='float32')

        # Fill X and Xmask
        for sentenceIndex in xrange(len(x)):
            # Insert sequence sentenceIndex in a column of matrix X
            sentenceLength = len(x[sentenceIndex])

            # This is mainly done for the mask not to include two EOS
            if x[sentenceIndex][sentenceLength - 1] == source_language.eos_index:
                sentenceLength -= 1

            realLength = min(mx, sentenceLength)
            X[:realLength, sentenceIndex] = x[sentenceIndex][:realLength]

            # Mark the end of phrase
            if sentenceLength < mx:
                X[sentenceLength:, sentenceIndex] = source_language.eos_index

            # Initialize Xmask column with ones in all positions that
            # were just set in X
            Xmask[:sentenceLength, sentenceIndex] = 1.
            if sentenceLength < mx:
                Xmask[sentenceLength, sentenceIndex] = 1.

        # Fill Y and Ymask in the same way as X and Xmask in the previous loop
        for sentenceIndex in xrange(len(y)):
            sentenceLength = len(y[sentenceIndex])

            # This is mainly done for the mask not to include two EOS
            if y[sentenceIndex][sentenceLength - 1] == target_language.eos_index:
                sentenceLength -= 1

            realLength = min(my, sentenceLength)
            Y[:realLength, sentenceIndex] = y[sentenceIndex][:realLength]
            if sentenceLength < my:
                Y[sentenceLength:, sentenceIndex] = target_language.eos_index
            Ymask[:sentenceLength, sentenceIndex] = 1.

            if sentenceLength < my:
                Ymask[sentenceLength, sentenceIndex] = 1.

        null_inputs = np.zeros(X.shape[1])

        # We say that an input pair is valid if both:
        # - either source sequence or target sequence is non-empty
        # - source sequence and target sequence have null_sym ending
        # Why did not we filter them earlier?
        for sentenceIndex in xrange(X.shape[1]):
            if np.sum(Xmask[:,sentenceIndex]) == 0 and np.sum(Ymask[:,sentenceIndex]) == 0:
                null_inputs[sentenceIndex] = 1
            if Xmask[-1,sentenceIndex] and X[-1,sentenceIndex] != source_language.eos_index:
                null_inputs[sentenceIndex] = 1
            if Ymask[-1,sentenceIndex] and Y[-1,sentenceIndex] != target_language.eos_index:
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
        X[X >= source_language.n_sym] = source_language.unk_index
        Y[Y >= target_language.n_sym] = target_language.unk_index

    # -- end else

    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask


