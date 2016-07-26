
import numpy
import logging

from experiments.nmt.language import create_padded_batch


class UnbufferedDataIterator:

    def __init__(self, source_file, target_file, state,
                 w2i_src, w2i_trg, i2w_src, i2w_trg,
                 num_sentences, sequence_length,
                 max_samples=None):
        self.peeked = None
        self.is_exhausted = False
        self.num_sentences = num_sentences
        self.sequence_length = sequence_length
        self.num_read = 0
        self.max_samples = max_samples
        self._keepNext = False
        self.state = state
        self.i2w_src = i2w_src
        self.i2w_trg = i2w_trg
        self.w2i_src = w2i_src
        self.w2i_trg = w2i_trg

        self.source_file_handler = open(source_file, 'r')
        self.target_file_handler = open(target_file, 'r')

    def start(self, start_offset):
        pass

    def peekBatch(self):
        assert self.peeked is None
        assert not self.is_exhausted

        self.peeked = [None] * 2
        self.peeked[0] = []
        self.peeked[1] = []

        for sentenceIndex in xrange(self.num_sentences):
            self.peeked[0].append(self.getSourceSentence())
            if self.peeked[0][-1] == '':
                # EOF reached
                # Cannot have happened before entering the loop
                self.peeked[0].pop()
                break

            self.peeked[1].append(self.getTargetSentence())

        self.is_exhausted = self.endOfSourceFile()
        self.num_read += 1
        if self.num_read == self.max_samples:
            self.is_exhausted = True

        return self.peeked

    def keepNext(self):
        # Used to train multiple iterations with the same sample
        self._keepNext = True

    def next(self):
        if self.peeked is None:
            self.peekBatch()

        sourceBatch = self.peeked[0]
        targetBatch = self.peeked[1]

        if self._keepNext:
            self._keepNext = False
        else:
            self.peeked = None

        batch = create_padded_batch([numpy.asarray(sourceBatch, dtype='int64')],
                                    [numpy.asarray(targetBatch, dtype='int64')],
                                    self.sequence_length, True,
                                    self.state, return_dict=True)
        if batch is None:
            logging.debug('Source batch:')
            logging.debug(sourceBatch)
            logging.debug('\nTarget batch:')
            logging.debug(targetBatch)
            raise Exception('create_padded_batch returned None, try increasing seqlen in the state to at least %d' %
                            (max(len(sourceBatch), len(targetBatch)) + 1))

        return batch

    def endOfSourceFile(self):
        currentPosition = self.source_file_handler.tell()
        nextline = self.source_file_handler.readline()
        if nextline == '':
            return True
        else:
            self.source_file_handler.seek(currentPosition)
            return False

    def indicesToWords(self, seq, i2w):
        try:
            text = ' '.join([i2w[i] for i in seq])
        except LookupError as exception:
            raise LookupError("Index out of dictionary of language: %d" % (exception.args[0]))
        return text

    def wordsToIndices(self, seq, w2i, i2w, unk_index=1, eos_index=0):
        # For compatibility with legacy function parse_input
        # this function adds an end-of-line word.
        words = seq.split()
        result = numpy.zeros(len(words) + 1, dtype='int64')

        for index, word in enumerate(words):
            result[index] = w2i.get(words[index], unk_index)
        result[-1] = eos_index
        string = self.indicesToWords(result, i2w)
        return result, string

    def getSourceSentence(self):
        return self.wordsToIndices(self.source_file_handler.readline(), self.w2i_src, self.i2w_src,
                                   unk_index=self.state['unk_sym_source'],
                                   eos_index=self.state['null_sym_source'])[0]

    def getTargetSentence(self):
        return self.wordsToIndices(self.target_file_handler.readline(), self.w2i_trg, self.i2w_trg,
                                   unk_index=self.state['unk_sym_target'],
                                   eos_index=self.state['null_sym_target'])[0]

    def exhausted(self):
        return self.is_exhausted

    def __iter__(self):
        return self

    def reset(self):
        self.source_file_handler.seek(0)
        self.target_file_handler.seek(0)
        self.is_exhausted = False
        self.num_read = 0
        self.peeked = None
