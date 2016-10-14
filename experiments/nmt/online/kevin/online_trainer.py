import logging

class Trainer:

    def __init__(self, model, dataIterator, trainingAlgorithm,
                 maxEpochs, outputPrefix,
                 hypothesesPolicy = 'last_epoch',
                 sampler = None,
                 modelSavingPolicy = 'last_epoch'):
        """
        Available policies for both model saving and hypotheses:
        always, never, last_epoch, `number` (meaning every N epochs)
        """
        self.model = model
        self.dataIterator = dataIterator
        self.trainingAlgorithm = trainingAlgorithm
        self.currentEpoch = 1
        self.maxEpochs = maxEpochs

        self.sampler = sampler
        self.hypothesesPolicy = hypothesesPolicy
        if hypothesesPolicy != 'never' and sampler is None:
            raise Exception('Hypotheses policy is %s but no sampler was provided.' % hypothesesPolicy)
        elif hypothesesPolicy == 'never' and sampler is not None:
            logging.warning('Unnecessary sampler was provided to Trainer.')

        self.modelSavingPolicy = modelSavingPolicy
        self.outputPrefix = outputPrefix
        print "Training algo:", self.trainingAlgorithm
    def train(self):
        while self.currentEpoch <= self.maxEpochs:
            self.trainOneEpoch()

    def trainOneEpoch(self):
        self.dataIterator.reset()
        self.currentEpochHypotheses = []

        logging.debug('Starting new epoch')
        while not self.dataIterator.exhausted():
            # [0]: Only source batch
            current_batch = self.dataIterator.peekBatch()[0]

            # In online training, we may want to get hypotheses for
            # the input.
            if self.shouldComputeHypotheses():
                self.computeHypotheses(current_batch)

            lastHypothesis = None
            if len(self.currentEpochHypotheses) != 0:
                lastHypothesis = self.currentEpochHypotheses[-1]

            self.runTrainingAlgorithm(lastHypothesis)


        if self.shouldSaveModel():
            self.saveModel()

        if self.shouldComputeHypotheses():
            self.saveHypotheses()

        self.currentEpoch += 1

    def shouldComputeHypotheses(self):
        isInt = isinstance(self.hypothesesPolicy, (int, long))
        isStr = isinstance(self.hypothesesPolicy, (str, unicode))

        if self.hypothesesPolicy == 'last_epoch':
            return self.currentEpoch == self.maxEpochs
        elif self.hypothesesPolicy == 'always':
            return True
        elif self.hypothesesPolicy == 'never':
            return False
        elif isInt or (isStr and self.hypothesesPolicy.isdigit()):
            # This is: compute every N epochs
            return self.currentEpoch % int(self.hypothesesPolicy) == 0
        else:
            # This 'if' is here to prevent a flood of warnings:
            if not self.invalidHypothesesPolicyWarning:
                logging.warning('Hypotheses policy %s is not a valid option. "never" is assumed.' % self.hypothesesPolicy)
                self.invalidHypothesesPolicyWarning = True
            return False

    def computeHypotheses(self, inputBatch):
        for sentence in inputBatch:
            translation = self.sampler.translate(sentence)
            if translation is None:
                logging.debug('Translation failed. Empty sentence is inserted.')
                translation = ''
            else:
                logging.info('Hypothesis: %s' % translation)
            self.currentEpochHypotheses.append(translation)

    def runTrainingAlgorithm(self, hypothesis):
        # Training algorithms define __call__ method, which accepts
        # hypothesis as single argument.

        # The only algorithm that requires it is PA, the others ignore
        # the parameter.
        print "hypothesis", hypothesis
        self.trainingAlgorithm(hypothesis)

    def shouldSaveModel(self):
        isInt = isinstance(self.modelSavingPolicy, (int, long))
        isStr = isinstance(self.modelSavingPolicy, (str, unicode))
        if self.modelSavingPolicy == 'never':
            return False
        elif self.modelSavingPolicy == 'always':
            # This is a bad idea
            return True
        elif self.modelSavingPolicy == 'last_epoch':
            return self.currentEpoch == self.maxEpochs
        elif isInt or (isStr and self.modelSavingPolicy.isdigit()):
            # This is: save every N epochs
            return self.currentEpoch % int(self.modelSavingPolicy) == 0
        else:
            # This 'if' is here to prevent a flood of warnings:
            if not self.invalidModelSavingPolicyWarning:
                logging.warning('Model saving policy %s is not a valid option. "never" is assumed.' % self.modelSavingPolicy)
                self.invalidModelSavingPolicyWarning = True
            return False

    def saveModel(self):
        logging.debug('Saving model...')
        self.model.save(self.getOutputModelPath())

    def getOutputModelPath(self):
        if self.currentEpoch >= self.maxEpochs:
            return self.outputPrefix + 'model'
        else:
            return (self.outputPrefix + 'model_epoch_%d') % self.currentEpoch

    def saveHypotheses(self):
        file = open(self.getOutputHypothesesPath(), 'w')
        for hypothesis in self.currentEpochHypotheses:
            file.write(hypothesis)
            file.write("\n")
        file.close()

    def getOutputHypothesesPath(self):
        if self.currentEpoch >= self.maxEpochs:
            return self.outputPrefix + 'hypotheses'
        else:
            return (self.outputPrefix + 'hypotheses_epoch_%d') % self.currentEpoch
