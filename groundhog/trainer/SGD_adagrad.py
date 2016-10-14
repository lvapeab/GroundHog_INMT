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
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim,
                                                dtype=x.dtype),
                                    name=x.name) for x in model.inputs]
        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
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
        lr = TT.scalar('lr')
        self.prop_names = [x[0] for x in model.properties]
        self.prop_exprs = [x[1] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        inputs_replacement_list = zip(model.inputs, loc_data)


        accumulated_squared_gradients = [theano.shared(numpy.zeros(param.shape.eval(),
                                                                   dtype=param.dtype),
                                                       name = param.name + '_acc_grad')
                                         for param in model.params]

        output_values = []
        st = time.time()

        def shift_zeroes(x):
            # The formula would create NaN upon the presence of zeroes
            return x + (abs(x) < 1e-8) * 1e-8


        def nan_and_inf_to_zero(x):
            x = TT.switch(TT.isnan(x), 0.0, x)
            return TT.switch(TT.isinf(x), 1e8, x)


        no_nan_or_inf_gradients = [nan_and_inf_to_zero(gradient) for gradient in self.gs]

        # Compute G_{i,i}
        accumulated_squared_gradients_update_list = [(acc_gradient, acc_gradient + gradient**2)
                                        for acc_gradient, gradient in
                                        zip(accumulated_squared_gradients, no_nan_or_inf_gradients)]

        # θ{t+1,i}=θ{t,i}−\div{η}{\sqrt{G_{t,ii}+ϵ}}*g_{t,i}
        weight_update_list = [(weight, weight - lr / TT.sqrt(TT.inc_subtensor(G_t[1][:],  1e-8)+TT.pow(gradient, 2))
                               * gradient) for weight, gradient, G_t
                              in zip(model.params, no_nan_or_inf_gradients, accumulated_squared_gradients_update_list)]

        updates = weight_update_list + accumulated_squared_gradients_update_list
        print 'Compiling update function'
        self.lr = numpy.float32(state['lr'])
        print '\t > Using a learning rate of', self.lr
        self.update_fn = theano.function(
            [lr], output_values, name='update_function',
            allow_input_downcast=True,updates = updates,
            profile=self.state['profile'],
            )

        print 'took', time.time() - st

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

        g_st = time.time()
        rvals = self.update_fn(self.lr)
        g_ed = time.time()
        whole_time = time.time() - self.step_timer
        if self.step % self.state['trainFreq'] == 0:
            msg = '.. iter %s'
            vals = [self.step]
            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
            msg += ' step time %s whole time %s lr %.2e'
            vals += [print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     float(self.lr)]
            print msg % tuple(vals)
        self.step += 1
        ret = dict([('lr', float(self.lr)),
                       ('time_step', float(g_ed - g_st)),
                       ('whole_time', float(whole_time))]+zip(self.prop_names, rvals))
        return ret