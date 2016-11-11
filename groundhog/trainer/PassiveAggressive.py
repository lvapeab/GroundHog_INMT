# -*- coding: utf-8 -*-
"""
 Iterative implementation of the Passive Aggressive algorithm (or an approximation to)
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
                 tolerance=0.,
                 C=1.):

        #####################################
        # Step 0. Constructs shared variables
        #####################################

        assert tolerance >= 0, 'Tolerance value should always be >= 0'

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
        self.gdata = [theano.shared(np.zeros( (2,)*x.ndim, dtype=x.dtype), name=x.name) for x in model.inputs]
        self.gs = [theano.shared(np.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX), name=p.name)
                   for p in model.params]
        self.tolerance = tolerance
        self.C = C
        self.eps = 1e-4
        if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################

        print 'Constructing grad function'
        loc_data = self.gdata
        self.prop_names = [x[0] for x in model.properties]
        self.prop_exprs = [x[1] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]

        inputs_replacement_list = zip(model.inputs, loc_data)  # = [(y, y), (y_mask, y_mask), (x, x), (x_mask, x_mask)]

        rval = theano.clone(model.param_grads + self.update_rules + self.prop_exprs + [model.train_cost],
                            replace=zip(model.inputs, loc_data))
        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]
        """
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
        """

        outputs = []

        hypothesis = T.lmatrix('hypothesis')
        hyp_mask = T.matrix('hyp_mask')
        inputDict = {}
        gdataDict = {}
        for inputVar in model.inputs:
            inputDict[inputVar.name] = inputVar
        for inputVar in self.gdata:
            gdataDict[inputVar.name] = inputVar
        hypothesis_replacement_list = filter(lambda (x, y): x.name != 'y' and  x.name != 'y_mask',
                                             inputs_replacement_list) # Get 'x' and 'x_mask'
        hypothesis_replacement_list.append((inputDict['y'], hypothesis))
        hypothesis_replacement_list.append((inputDict['y_mask'], hyp_mask))
        target_replacement_list = inputs_replacement_list

        #print "hypothesis_replacement_list", hypothesis_replacement_list # = [(x, x), (x_mask, x_mask), (y, hypothesis), (y_mask, hyp_mask)]
        #print "target_replacement_list", target_replacement_list # = [(y, y), (y_mask, y_mask), (x, x), (x_mask, x_mask)]
        #print "gdata", self.gdata # = [y, y_mask, x, x_mask]
        # yCost = theano.clone(model.cost_layer.cost, replace=target_replacement_list)
        # hCost = theano.clone(model.cost_layer.cost, replace=hypothesis_replacement_list)
        #loss = theano.clone(model.cost_layer.cost, replace=hypothesis_replacement_list)#theano.clone(model.cost_layer.cost, replace=target_replacement_list) #- theano.clone(model.cost_layer.cost, replace=hypothesis_replacement_list)

        loss = T.fscalar('loss')
        loss = theano.printing.Print('loss=')(loss)
        # hCost - yCost
        #loss = theano.clone(model.cost_layer.cost) - theano.clone(model.cost_layer.cost)
        lossGrads = [T.grad(loss, weight, return_disconnected='zero') for weight in model.params]
        new_params = [p +  l * g  for p, l, g in
                      zip(model.params, lossGrads, self.gs)]
        loss2 = loss + 1.
        print 'Compiling update function'
        self.update_fn = theano.function(
            [hypothesis, hyp_mask, loss],
            [loss2], name='update_function',
            allow_input_downcast=True,
            updates = zip(model.params, new_params),
            profile=self.state['profile'])
        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + ['cost', 'time_step', 'whole_time']
        print 'End compiling PA algorithm'

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

        self.loss = hyp_prob - ref_prob
        rvals = []

        if self.loss <= self.tolerance:
            print ref_prob,  "<", hyp_prob, "!"
            print "NOW we should launch update fn!"
            print "(hypothesis: ", hypothesis,")"
            #self.train_fn()
            rvals = self.update_fn(np.asarray([hyp_indices]), np.asarray([hyp_mask]), self.loss)
        else:
            try:
                assert np.all(np.equal(trg_seq, hyp_indices)), '"' + hypothesis + '" is not equal to reference and has ' \
                                                        'more probability!! \n' \
                                                            '\t Hypothesis probability: '+str(hyp_prob)+ '\n' \
                                                            '\t Reference probability: ' + str(ref_prob)  + '\n'
            except:
                pass
            return

        #if len(rvals) > 0:
        #    self.print_trace(rvals)
        #    # rvals[0] must always be loss
        #    loss = rvals[0]
        #    if abs(loss) < 1:
        #        return
