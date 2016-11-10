# -*- coding: utf-8 -*-


""" Adagrad learning algorithm. For a nice explanation of the algorithm, see: http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad
Adagrad [1] is an algorithm for gradient-based optimization
that adapts the learning rate to the parameters, performing
larger updates for infrequent and smaller updates for frequent parameters.

We set g_{t,i} to be the gradient of the objective function w.r.t. to the parameter θi at time step t
gt,i=∇θJ(θi)
Adagrad modifies the general learning rate η at each time step t for every parameter θi based
on the past gradients that have been computed for θi:
θ{t+1,i}=θ{t,i}−\div{η}{\sqrt{G_{t,ii}+ϵ}}*g_{t,i}

Gt∈ℝ^{d×d}: Diagonal matrix where each diagonal element i,i is
the sum of the squares of the gradients w.r.t. θi up to time step t

[1]: Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
 Journal of Machine Learning Research, 12, 2121–2159. (http://jmlr.org/papers/v12/duchi11a.html)
"""

__docformat__ = 'restructedtext en'
__authors__ = ("Alvaro Peris "
               "Kevin Montalva")
__contact__ = "Alvaro Peris <lvapeab@gmail>"

import numpy
import logging
import time

import theano
import theano.tensor as TT

from groundhog.utils import print_time

logger = logging.getLogger(__name__)
class AdaGrad(object):


    def __init__(self,
                 model,
                 state,
                 data):

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
        self.gdata = [theano.shared(numpy.zeros((2,)*x.ndim,
                                                dtype=x.dtype), name=x.name) for x in model.inputs]
        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name)
                   for p in model.params]

        # G_{i,i}
        self.accumulated_squared_gradients = [theano.shared(numpy.zeros(param.shape.eval(), dtype=param.dtype),
                                                       name = param.name + '_acc_grad')
                                         for param in model.params]
        self.eps = 1e-4
        if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################
        print 'Constructing grad function'
        loc_data = self.gdata
        lr = TT.scalar('lr')
        self.prop_names = [x[0] for x in model.properties]
        self.prop_exprs = [x[1] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        #inputs_replacement_list = zip(model.inputs, loc_data)
        rval = theano.clone(model.param_grads + self.update_rules + \
                            self.prop_exprs + [model.train_cost],
                            replace=zip(model.inputs, loc_data))
        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]

        norm_gs = sum(TT.sum(x**2)
            for x,p in zip(gs,
                           self.model.params)
                      if p not in self.model.exclude_params_for_norm)
        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []
            for g,p in zip(gs,self.model.params):
                if p not in self.model.exclude_params_for_norm:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1)*p,
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


        # Compute G_{i,i}
        accumulated_squared_gradients_update_list = [(acc_gradient, acc_gradient + gradient**2)
                                        for acc_gradient, gradient in
                                        zip(self.accumulated_squared_gradients, self.gs)]

        # θ{t+1,i}=θ{t,i}−\div{η}{\sqrt{G_{t,ii}+ϵ}}*g_{t,i}
        updates = [(weight, weight - lr * gradient / TT.sqrt(G_t[1]+TT.pow(gradient, 2)+1e-8))
                              for weight, G_t, gradient in zip(model.params, accumulated_squared_gradients_update_list, self.gs)]

        print 'Compiling update function'
        self.lr = numpy.float32(state['lr'])
        print '\t > Using a learning rate of', self.lr
        self.update_fn = theano.function(
            [lr], [], name='update_function',
            allow_input_downcast=True,updates=updates,
            profile=self.state['profile'],
            )

        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                ['cost',
                 'time_step',
                 'whole_time']

    def __call__(self, _):
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
        rvals = self.train_fn()
        g_st = time.time()
        self.update_fn(self.lr)
        g_ed = time.time()
        whole_time = time.time() - self.step_timer
        self.state['lr'] = float(self.lr)

        if self.step % self.state['trainFreq'] == 0:
            msg = '.. iter %s'
            vals = [self.step]
            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
            msg += ' step time %s whole time %s lr %.2e'
            vals += [print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     float(self.lr)]
        self.step += 1
        ret = dict([('lr', float(self.lr)),
                       ('time_step', float(g_ed - g_st)),
                       ('whole_time', float(whole_time))]+zip(self.prop_names, rvals))
        return ret