# -*- coding: utf-8 -*-


""" Iterative implementation of the Passive Aggressive algorithm (or an approximation to)
"""

__docformat__ = 'restructedtext en'
__authors__ = ("Alvaro Peris ")
__contact__ = "Alvaro Peris <lvapeab@gmail>"

import numpy
import logging
import time

import theano
import theano.tensor as T

from groundhog.utils import print_time, nan_and_inf_to_zero, shift_zeroes

logger = logging.getLogger(__name__)
class PassiveAggressive(object):


    def __init__(self,
                 model,
                 state,
                 data,
                 word_probs,
                 tolerance=1.):

        #####################################
        # Step 0. Constructs shared variables
        #####################################

        bs = state['bs']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])

        self.add_noise = state['weight_noise']
        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.step_timer = time.time()
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim, dtype=x.dtype), name=x.name)
                      for x in model.inputs]

        self.eps = 1e-4
        if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################

        print 'Constructing grad function'
        loc_data = self.gdata
        lr = T.scalar('lr')
        self.prop_names = [x[0] for x in model.properties]
        self.prop_exprs = [x[1] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        self.iterationInitialParams = [theano.shared(numpy.array(param.get_value(),
                                                                 dtype = param.dtype),
                                                     name = "%s_iteration_initial" % param.name)
                                       for param in model.params]
        inputs_replacement_list = zip(model.inputs, loc_data)

        rval = theano.clone(model.param_grads + self.update_rules + self.prop_exprs + [model.train_cost],
                            replace=inputs_replacement_list)
        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        parameter_gradients = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]

        inputDict = {}
        gdataDict = {}
        for inputVar in model.inputs:
            inputDict[inputVar.name] = inputVar
        for inputVar in self.gdata:
            gdataDict[inputVar.name] = inputVar

        inputs_replacement_list = zip(model.inputs, loc_data)

        target_replacement_list = inputs_replacement_list[:]

        hypothesis = T.lmatrix('hypothesis')
        hyp_mask = T.matrix('hyp_mask')
        hypothesis_replacement_list = filter(lambda (x, y): x.name != 'y' and \
                                             x.name != 'y_mask',
                                             inputs_replacement_list)
        hypothesis_replacement_list.append((inputDict['y'],
                                            hypothesis))
        hypothesis_replacement_list.append((inputDict['y_mask'],
                                            hyp_mask))

        parameter_gradients = theano.clone(model.param_grads, replace=inputs_replacement_list)
        # word_probs = word_probs * inputDict['y_mask'] + (1 - inputDict['y_mask'])

        y_probs = theano.clone(word_probs, replace=target_replacement_list)
        hyp_probs = theano.clone(word_probs, replace=hypothesis_replacement_list)
        # word_probs = word_probs + (word_probs < 1e-7) * 1e-7

        yCost = T.sum(-T.log(y_probs))
        hypCost = T.sum(-T.log(hyp_probs))

        self.outputNames = []
        outputs = []

        loss = hypCost - yCost
        self.loss = loss
        loss = theano.clone(model.cost_layer.cost, replace = hypothesis_replacement_list) - \
               theano.clone(model.cost_layer.cost, replace = target_replacement_list)
        # loss must be first output
        self.outputNames.append('loss')
        outputs.append(loss)

        lossGrads = [T.grad(loss, weight) for weight in model.params]

        lossGradsNorm = T.sqrt(T.sum([T.sum(T.sqr(grad)) for grad in lossGrads]))

        k = 1
        if k == 2:
            # For some reason, 2 * tolerance is cast to float64, even
            # if theano.config.floatX is 32, this fixes it:
            doubleTolerance = numpy.array(2, dtype=theano.config.floatX) * tolerance
            one = numpy.array(1, dtype = theano.config.floatX)
            lagrangeMultiplier = - one / ((lossGradsNorm ** 2) - one / doubleTolerance)

            updateList = [-lagrangeMultiplier * lossGrad
                          for lossGrad in lossGrads]
            self.outputNames.append('Lambda')
            outputs.append(lagrangeMultiplier)
            self.outputNames.append('lossGradsNorm')
            outputs.append(lossGradsNorm)
        elif k == 1:
            lagrangeMultiplier1 = T.min([tolerance, -loss / (lossGradsNorm ** 2)])
            lagrangeMultiplier2 = tolerance - lagrangeMultiplier1
            updateList = [-lagrangeMultiplier1
                          * lossGrad
                          for lossGrad in lossGrads]

            self.outputNames.append('lossGradsNorm')
            outputs.append(lossGradsNorm)
        else:
            raise Exception('Unsupported k: %d' % k)

        updateNorm = T.sqrt(T.sum([T.sum(T.sqr(update)) for update in updateList]))
        self.outputNames.append('updateNorm')
        outputs.append(updateNorm)

        print 'Compiling training function'
        st = time.time()

        """
        # Average of weights in absolute value was 0.0002.
        replacementConstant = 1e-1
        updateList = [T.maximum(-replacementConstant,
                                 T.minimum(replacementConstant, update))
                      for update in updateList]
        """

        self.update_fn = theano.function(
            [hypothesis, hyp_mask],
            outputs,
            name='update_function',
            allow_input_downcast=True,
            profile=self.state['profile'],
            updates = [(weight, initWeight + update)
                       for weight, initWeight, update
                       in zip(model.params,
                              self.iterationInitialParams,
                              updateList)])

        print 'took', time.time() - st

        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                ['cost',
                 'time_step',
                 'whole_time']

    def setIterationInitialWeights(self):
        for initialParam, param in zip(self.iterationInitialParams,
                                       self.model.params):
            # Any way to do this more efficiently?
            initialParam.set_value(param.get_value())

    def __call__(self, hypothesis):
        """
        Ignored parameter: hypothesis.
        """
        batch = self.data.next()

        """
        # Perturb the data (! and the model)
        if self.add_noise:
            if isinstance(batch, dict):
                batch = self.model.perturb(**batch)
            else:
                batch = self.model.perturb(*batch)
        """
        # Load the dataset into GPU
        # Note: not the most efficient approach in general, as it involves
        # each batch is copied individually on gpu
        if isinstance(batch, dict):
            for gdata in self.gdata:
                # print batch[gdata.name]
                gdata.set_value(batch[gdata.name], borrow=True)
        else:
            for gdata, data in zip(self.gdata, batch):
                gdata.set_value(data, borrow=True)


        hyp_nums = self.model.target_language.wordsToIndices(hypothesis)[0]
        hyp = numpy.ones((len(hyp_nums), 1),
                         dtype = 'int64')
        hyp[:, 0] = hyp_nums
        hyp_mask = numpy.ones((hyp_nums.shape[0], 1),
                              dtype = theano.config.floatX)

        self.setIterationInitialWeights()

        g_st = time.time()
        rvals = self.update_fn(self.gdata, float(self.lr))
        if len(rvals) > 0:
            self.output(rvals)
            # rvals[0] must always be loss
            loss = rvals[0]
            if abs(loss) < 1:
                return
