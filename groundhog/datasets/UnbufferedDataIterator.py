
import numpy
import logging

from experiments.nmt.language import create_padded_batch

class UnbufferedDataIterator():

    def __init__(self, source_file, target_file,
                 source_language, target_language,
                 num_sentences, sequence_length,
                 max_samples=None,
                 debug=None):
        self.peeked = None
        self.is_exhausted = False
        self.source_language = source_language
        self.target_language = target_language
        self.num_sentences = num_sentences
        self.sequence_length = sequence_length
        self.num_read = 0
        self.max_samples = max_samples
        self._keepNext = False

        self.source_file_handler = open(source_file, 'r')
        self.target_file_handler = open(target_file, 'r')
        self.debug = debug
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
                                    self.source_language,
                                    self.target_language, return_dict=True)
        if batch is None:
            logging.debug('Source batch:')
            logging.debug(sourceBatch)
            logging.debug('\nTarget batch:')
            logging.debug(targetBatch)
            raise Exception('create_padded_batch returned None, try increasing seqlen in the state to at least %d' %
                            (max(len(sourceBatch), len(targetBatch)) + 1))
        if self.debug:
            logging.debug('Source batch:')
            logging.debug(sourceBatch)
            logging.debug('\nTarget batch:')
            logging.debug(targetBatch)
        return batch

    def endOfSourceFile(self):
        currentPosition = self.source_file_handler.tell()
        nextline = self.source_file_handler.readline()
        if nextline == '':
            return True
        else:
            self.source_file_handler.seek(currentPosition)
            return False

    def getSourceSentence(self):
        return self.source_language.wordsToIndices(self.source_file_handler.readline())[0]

    def getTargetSentence(self):
        return self.target_language.wordsToIndices(self.target_file_handler.readline())[0]

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
