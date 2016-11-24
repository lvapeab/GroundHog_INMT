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
                 word_probs,
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
        self.gdata = [theano.shared(np.zeros( (2,)*x.ndim,
                                              dtype=x.dtype), name=x.name) for x in model.inputs]
        self.gs = [theano.shared(np.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX), name=p.name)
                   for p in model.params]
        self.tolerance = tolerance
        self.C = C
        self.eps = 1e-4
        self.lr = np.float32(state['lr'])

        if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################

        print 'Constructing grad function'
        st0 = time.time()
        loc_data = self.gdata

        self.prop_names = [x[0] for x in model.properties]
        self.prop_exprs = [x[1] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        inputs_replacement_list = zip(model.inputs, loc_data)
        """
        rval = theano.clone(model.param_grads + self.update_rules + self.prop_exprs + [model.train_cost],
                            replace=inputs_replacement_list)
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
        """

        st = time.time()
        self.train_fn = theano.function(
            [], [], name='train_function',
            givens=zip(model.inputs, loc_data),
            profile=self.state['profile'])
        print 'took', time.time() - st0, 'seconds'
        ##########################################################
        #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
        ##########################################################
        #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
        ##########################################################

        x = T.lmatrix('x')
        x_mask = T.matrix('x_mask')
        y = T.lmatrix('y')
        y_mask = T.matrix('y_mask')
        hypothesis = T.lmatrix('hypothesis')
        hyp_mask = T.matrix('hyp_mask')

        inputs_replacement_list = zip(model.inputs, loc_data)

        hypothesis_replacement_list = [('x', x),('x_mask', x_mask),
                                        ('y', hypothesis),('y_mask', hyp_mask)]

        target_replacement_list = {'x': x,'x_mask': x_mask, 'y': y,'y_mask': y_mask }

        #loss =theano.clone(model.cost_layer.cost, replace = hypothesis_replacement_list)  \
        #                    - theano.clone(model.cost_layer.cost, replace = target_replacement_list)
        y_probs = theano.clone(model.cost_layer.get_cost, replace=target_replacement_list)
        loss = y_probs

        print "Computing loss grads"
        st = time.time()
        lossGrads = [T.grad(loss, weight, disconnected_inputs='ignore') for weight in model.params]

        print "loss grads computed"
        print 'took', time.time() - st, 'seconds'

        #new_params = [p + loss / lossGradsNorm * lossGrad for p, lossGrad, lossGradsNorm in zip(model.params, lossGrads, lossGradsNorms)]
        new_params = [p - 0*l for p, l in zip(model.params, lossGrads)]
        print 'Compiling update function'
        st = time.time()

        self.update_fn = theano.function(
            [x, x_mask, y, y_mask, hypothesis, hyp_mask],
            y_probs, name='update_function',
            allow_input_downcast=True,
            updates=zip(model.params, new_params),
            givens=zip(model.inputs, loc_data))
        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + ['cost', 'time_step', 'whole_time']
        print 'took', time.time() - st, 'seconds'
        print 'End compiling PA algorithm. Full compilation took', 'took', time.time() - st0, 'seconds'

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
        #print "RVALS:", rvals

        src_seq = batch['x'].flatten()
        trg_seq = batch['y'].flatten()
        trg_mask = batch['y_mask'].flatten()
        src_mask = batch['x_mask'].flatten()

        ref_word_probs = self.probs_computer(src_seq, trg_seq)
        hyp_indices, _ = self.model.target_language.wordsToIndices(hypothesis)
        hyp_mask = np.ones((hyp_indices.shape[0]), dtype=theano.config.floatX)
        hyp_word_probs = self.probs_computer(src_seq, hyp_indices)

        ref_prob = ref_word_probs.flatten().sum()
        hyp_prob = hyp_word_probs.flatten().sum()
        self.train_fn()
        self.loss = ref_prob - hyp_prob
        rvals = []
        if self.loss <= self.tolerance:
            print "loss:", self.loss, "> 0! (" ,ref_prob,  "(ref_prob) >", hyp_prob, "(hyp_prob))"
            print "NOW we should launch update fn!"
            print "(hypothesis: ", hypothesis,")"
            rvals = self.update_fn(np.asarray([src_seq]), np.asarray([src_mask]),
                                    np.asarray([trg_seq]), np.asarray([trg_mask]),
                                    np.asarray([hyp_indices]), np.asarray([hyp_mask]))#, self.lr)
            print "rvals=",rvals
            new_ref_word_probs = self.probs_computer(src_seq, trg_seq).flatten().sum()
            new_hyp_word_probs = self.probs_computer(src_seq, hyp_indices).flatten().sum()
            print "After update_fn: ref_prob", new_ref_word_probs, "; hyp prob: ", new_hyp_word_probs
        else:
            try:
                assert np.all(np.equal(trg_seq, hyp_indices)), '"' + hypothesis + '" is not equal to reference and has ' \
                                                        'more probability!! \n' \
                                                            '\t Hypothesis probability: '+str(hyp_prob)+ '\n' \
                                                            '\t Reference probability: ' + str(ref_prob)  + '\n'
            except:
                pass
            return

