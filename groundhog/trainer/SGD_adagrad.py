
import numpy
import logging
import time

import theano
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import print_time, print_mem, const

logger = logging.getLogger(__name__)

class AdaGrad(object):
    """
    
    """
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
        self.eps = 1e-4

        if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################

        print 'Constructing grad function'
        loc_data = self.gdata
        self.prop_exprs = [x[1] for x in model.properties]
        self.prop_names = [x[0] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]

        inputs_replacement_list = zip(model.inputs, loc_data)

        parameter_gradients = theano.clone(model.param_grads,
                                           replace = inputs_replacement_list)
        accumulated_squared_gradients = [theano.shared(numpy.zeros(param.shape.eval(),
                                                                   dtype=param.dtype),
                                                       name = param.name + '_acc_grad')
                                         for param in model.params]

        learning_rate = numpy.float32(state['lr'])

        output_values = []

        print 'Compiling training function'
        st = time.time()

        def shift_zeroes(x):
            # The formula would create NaN upon the presence of zeroes
            return x + (abs(x) < 1e-7) * 1e-7

        def nan_and_inf_to_zero(x):
            x = TT.switch(TT.isnan(x), 0.0, x)
            return TT.switch(TT.isinf(x), 0.0, x)

        no_nan_or_inf_gradients = [nan_and_inf_to_zero(gradient) for gradient in parameter_gradients]

        weight_update_list = [(weight, weight - learning_rate *
                               gradient /
                               shift_zeroes(TT.sqrt(accumulated_squared_gradient + TT.pow(gradient, 2))))
                              for weight, gradient,
                              accumulated_squared_gradient in
                              zip(model.params, no_nan_or_inf_gradients,
                                  accumulated_squared_gradients)]

        accumulated_squared_gradients_update = [(acc_gradient,
                                                 acc_gradient + TT.pow(gradient, 2)) for
                                                acc_gradient, gradient in
                                                zip(accumulated_squared_gradients,
                                                    no_nan_or_inf_gradients)]

        self.update_fn = theano.function(
            [], output_values, name='update_function',
            allow_input_downcast=True,
            profile=self.state['profile'],
            updates = weight_update_list + accumulated_squared_gradients_update)
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
        print "Batch:", batch
        print "Batch", batch.keys()
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

        rvals = self.update_fn()
        if len(rvals) > 0:
            print rvals
