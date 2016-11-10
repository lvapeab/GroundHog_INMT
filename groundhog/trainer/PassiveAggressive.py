# -*- coding: utf-8 -*-


""" Iterative implementation of the Passive Aggressive algorithm (or an approximation to)
"""

__docformat__ = 'restructedtext en'
__authors__ = ("Alvaro Peris ")
__contact__ = "Alvaro Peris <lvapeab@gmail>"

import numpy as np
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
                 probs_computer,
                 tolerance=1.):

        #####################################
        # Step 0. Constructs shared variables
        #####################################

        bs = state['bs']
        self.model = model
        self.rng = np.random.RandomState(state['seed'])

        self.add_noise = state['weight_noise']
        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.probs_computer = probs_computer
        self.step_timer = time.time()
        self.gdata = [theano.shared(np.zeros( (2,)*x.ndim, dtype=x.dtype), name=x.name)
                      for x in model.inputs]
        self.gs = [theano.shared(np.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                 name=p.name)
                   for p in model.params]

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
        self.iterationInitialParams = [theano.shared(np.array(param.get_value(),
                                                                 dtype = param.dtype),
                                                     name = "%s_iteration_initial" % param.name)
                                       for param in model.params]
        inputs_replacement_list = zip(model.inputs, loc_data)
        rval = theano.clone(model.param_grads + self.update_rules + \
                            self.prop_exprs + [model.train_cost],
                            replace=zip(model.inputs, loc_data))
        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]

        norm_gs = sum(T.sum(x**2)
            for x,p in zip(gs,
                           self.model.params)
                      if p not in self.model.exclude_params_for_norm)
        if 'cutoff' in state and state['cutoff'] > 0:
            c = np.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * T.cast(loc_data[0].shape[0], 'float32')

            notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
            _gs = []
            for g,p in zip(gs,self.model.params):
                if p not in self.model.exclude_params_for_norm:
                    tmpg = T.switch(T.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       T.switch(notfinite, np.float32(.1)*p,
                           tmpg))
                else:
                    _gs.append(g)
            gs = _gs

        store_gs = [(s,g) for s,g in zip(self.gs, gs)]
        training_updates = store_gs + [(s[0], r) for s,r in zip(model.updates, rules)]
        print 'Compiling grad function'
        self.train_fn = theano.function(
            [], outs, name='train_function',
            updates = training_updates,
            givens = zip(model.inputs, loc_data),
            profile=self.state['profile'])


        inputDict = {}
        gdataDict = {}
        for inputVar in model.inputs:
            # y, y_mask, x, x_mask
            inputDict[inputVar.name] = inputVar

        for inputVar in self.gdata:
            # y, y_mask, x, x_mask
            gdataDict[inputVar.name] = inputVar

        target_replacement_list = inputs_replacement_list[:]
        print "input_replacement_list", inputs_replacement_list
        hypothesis = T.lvector('hypothesis')
        hyp_mask = T.vector('hyp_mask')
        #hypothesis_replacement_list = filter(lambda (x, y): x.name != 'y' and x.name != 'y_mask',
        #                                     inputs_replacement_list)
        #hypothesis_replacement_list.append((inputDict['y'], hypothesis))
        #hypothesis_replacement_list.append((inputDict['y_mask'], hyp_mask))
        #print "hypothesis_replacement_list", hypothesis_replacement_list

        #src_seq = T.lmatrix('src_seq')
        #trg_seq = T.lmatrix('trg_seq')
        #probs = probs_computer(src_seq, trg_seq)
        #self.probs_computer = theano.function([src_seq, trg_seq], probs, name='probs_computer')

        outputs = []

        """
        #print "word_probs", word_probs
        #y_probs = theano.clone(word_probs, replace=target_replacement_list)
        #hyp_probs = theano.clone(word_probs, replace=hypothesis_replacement_list)

        #yCost = T.sum(-T.log(y_probs))
        #hypCost = T.sum(-T.log(hyp_probs))

        self.outputNames = []
        outputs = []

        loss = hypCost - yCost

        self.loss = loss
        loss = theano.clone(model.cost_layer.cost, replace = hypothesis_replacement_list) - \
               theano.clone(model.cost_layer.cost, replace = target_replacement_list)
        loss = theano.printing.Print('loss')(loss)
        # loss must be first output
        #self.outputNames.append('loss')
        #outputs.append(loss)
        #print "outputs", outputs
        """


        """
        lossGrads = [T.grad(loss, weight) for weight in model.params]
        lossGradsNorm = T.sqrt(T.sum([T.sum(T.sqr(grad)) for grad in lossGrads]))

        k = 1
        if k == 2:
            # For some reason, 2 * tolerance is cast to float64, even
            # if theano.config.floatX is 32, this fixes it:
            doubleTolerance = np.array(2, dtype=theano.config.floatX) * tolerance
            one = np.array(1, dtype = theano.config.floatX)
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
            updateList = [-lagrangeMultiplier1 * lossGrad for lossGrad in lossGrads]
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

        #updates = [(weight, initWeight + update)
        #               for weight, initWeight, update
        #               in zip(model.params, self.iterationInitialParams, updateList)]
        new_params = [p - s * g for s, p, g in zip(model.params_grad_scale, model.params, self.gs)]
        updates = zip(model.params, new_params)
        self.update_fn = theano.function(
            [hypothesis, hyp_mask],
            outputs,
            name='update_function',
            allow_input_downcast=True,
            profile=self.state['profile'],
            updates = updates)

        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + ['cost', 'time_step', 'whole_time']


    def setIterationInitialWeights(self):
        for initialParam, param in zip(self.iterationInitialParams, self.model.params):
            # Any way to do this more efficiently?
            initialParam.set_value(param.get_value())


    def print_trace(self, rvals):
        lineWidth = 80
        cellWidth = lineWidth / len(rvals)
        cellTextPattern = '%' + str(cellWidth) + 's'
        cellDataPattern = '%' + str(cellWidth) + '.6f'
        textLinePattern = cellTextPattern * len(rvals)
        dataLinePattern = cellDataPattern * len(rvals)
        print textLinePattern % tuple(self.outputNames)
        print dataLinePattern % tuple(rvals)
        print


    def __call__(self, hypothesis):
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

        src_seq = batch['x'].flatten()
        trg_seq = batch['y'].flatten()

        ref_word_probs = self.probs_computer(src_seq, trg_seq)
        hyp_indices, _ = self.model.target_language.wordsToIndices(hypothesis)
        hyp_mask = np.ones((hyp_indices.shape[0]), dtype = theano.config.floatX)
        hyp_word_probs = self.probs_computer(src_seq, hyp_indices)

        ref_prob = ref_word_probs.flatten().sum()
        hyp_prob = hyp_word_probs.flatten().sum()

        if ref_prob < hyp_prob:
            print ref_prob,  "<", hyp_prob, "!"
            print "NOW we should launch update fn!"
            print "(hypothesis: ", hypothesis,")"
            self.setIterationInitialWeights()
            rvals = self.update_fn(hyp_indices, hyp_mask)
        else:
            assert np.all(np.equal(trg_seq, hyp_indices)), '"' + hypothesis + '" is not equal to reference and has ' \
                                                        'more probability!! \n' \
                                                            '\t Hypothesis probability: '+str(hyp_prob)+ '\n' \
                                                            '\t Reference probability: ' + str(ref_prob)  + '\n'
            return

        if len(rvals) > 0:
            self.print_trace(rvals)
            # rvals[0] must always be loss
            loss = rvals[0]
            if abs(loss) < 1:
                return
