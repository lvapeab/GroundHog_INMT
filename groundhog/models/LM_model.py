"""
Implementation of a language model class.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import numpy
import itertools
import logging

import cPickle as pkl

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import id_generator
from groundhog.layers.basic import Model

logger = logging.getLogger(__name__)

class LM_Model(Model):
    def  __init__(self,
                  cost_layer = None,
                  sample_fn = None,
                  valid_fn = None,
                  noise_fn = None,
                  clean_before_noise_fn = False,
                  clean_noise_validation=True,
                  weight_noise_amount = 0,
                  need_inputs_for_generating_noise=False,
                  source_language=None,
                  target_language=None,
                  character_level = False,
                  exclude_params_for_norm=None,
                  rng = None):
        """
        Constructs a model, that respects the interface required by the
        trainer class.

        :type cost_layer: groundhog layer
        :param cost_layer: the cost (last) layer of the model

        :type sample_fn: function or None
        :param sample_fn: function used to sample from the model

        :type valid_fn: function or None
        :param valid_fn: function used to compute the validation error on a
            minibatch of examples

        :type noise_fn: function or None
        :param noise_fn: function called to corrupt an input (that
            potentially will be denoised by the model)

        :type clean_before_noise_fn: bool
        :param clean_before_noise_fn: If the weight noise should be removed
            before calling the `noise_fn` to corrupt some input

        :type clean_noise_validation: bool
        :param clean_noise_validation: If the weight noise should be removed
            before calling the validation function

        :type weight_noise_amount: float or theano scalar
        :param weight_noise_amount: weight noise scale (standard deviation
            of the Gaussian from which it is sampled)

        :type source_language: Language
        :param source_language: Language object corresponding to the source
            language

        :type target_language: Language
        :param target_language: Language object corresponding to the target
            language

        :type need_inputs_for_generating_noise: bool
        :param need_inputs_for_generating_noise: flag saying if the shape of
            the inputs affect the shape of the weight noise that is generated at
            each step

        :type character_level: bool
        :param character_level: flag used when sampling, saying if we are
            running the model on characters or words

        :type excluding_params_for_norm: None or list of theano variables
        :param excluding_params_for_norm: list of parameters that should not
            be included when we compute the norm of the gradient (for norm
            clipping). Usually the output weights if the output layer is
            large

        :type rng: numpy random generator
        :param rng: numpy random generator

        """
        super(LM_Model, self).__init__(output_layer=cost_layer,
                                       sample_fn=sample_fn,
                                       target_language=target_language,
                                       source_language=source_language,
                                       rng=rng)
        if exclude_params_for_norm is None:
            self.exclude_params_for_norm = []
        else:
            self.exclude_params_for_norm = exclude_params_for_norm
        self.need_inputs_for_generating_noise=need_inputs_for_generating_noise
        self.cost_layer = cost_layer
        self.validate_step = valid_fn
        self.clean_noise_validation = clean_noise_validation
        self.noise_fn = noise_fn
        self.clean_before = clean_before_noise_fn
        self.weight_noise_amount = weight_noise_amount
        self.character_level = character_level

        # Possible optimization: Do only if we are going to train
        # Alternative: do it in the training algorithm (probably better)
        logger.debug("Get grads")
        self.param_grads = T.grad(cost_layer.cost.mean(), self.params)
        logger.debug("Got grads")

        self.valid_costs = ['cost','ppl']
        # Assume a single cost
        # We need to merge these lists
        state_below = self.cost_layer.state_below
        if hasattr(self.cost_layer, 'mask') and self.cost_layer.mask:
            num_words = T.sum(self.cost_layer.mask)
        else:
            num_words = T.cast(state_below.shape[0], 'float32')
        scale = getattr(self.cost_layer, 'cost_scale', numpy.float32(1))
        if not scale:
            scale = numpy.float32(1)
        scale *= numpy.float32(numpy.log(2))

        grad_norm = T.sqrt(sum(T.sum(x ** 2)
                               for x,p in zip(self.param_grads, self.params) if p not in
                               self.exclude_params_for_norm))
        new_properties = [
                ('grad_norm', grad_norm),
                ('log2_p_word', self.train_cost / num_words / scale),
                ('log2_p_expl', self.cost_layer.cost_per_sample.mean() / scale)]
        self.properties += new_properties

        if len(self.noise_params) >0 and weight_noise_amount:
            if self.need_inputs_for_generating_noise:
                inps = self.inputs
            else:
                inps = []
            self.add_noise = theano.function(inps,[],
                                             name='add_noise',
                                             updates = [(p,
                                                 self.trng.normal(shp_fn(self.inputs),
                                                     avg =0,
                                                     std=weight_noise_amount,
                                                     dtype=p.dtype))
                                                 for p, shp_fn in
                                                        zip(self.noise_params,
                                                         self.noise_params_shape_fn)],
                                            on_unused_input='ignore')
            self.del_noise = theano.function(inps,[],
                                             name='del_noise',
                                             updates=[(p,
                                                       T.zeros(shp_fn(self.inputs),
                                                               p.dtype))
                                                      for p, shp_fn in
                                                      zip(self.noise_params,
                                                          self.noise_params_shape_fn)],
                                            on_unused_input='ignore')
        else:
            self.add_noise = None
            self.del_noise = None


    def validate(self, data_iterator, train=False):
        cost = 0
        n_batches = 0
        n_steps = 0
        if self.del_noise and self.clean_noise_validation:
            if self.need_inputs_for_generating_noise:
                self.del_noise(**vals)
            else:
                self.del_noise()

        for vals in data_iterator:
            n_batches += 1

            if isinstance(vals, dict):
                val = vals.values()[0]
                if val.ndim ==3:
                    n_steps += val.shape[0]*val.shape[1]
                else:
                    n_steps += val.shape[0]

                _rvals = self.validate_step( **vals)
                cost += _rvals
            else:
                # not dict
                if vals[0].ndim ==3:
                    n_steps += vals[0].shape[0]*vals[1].shape[1]
                else:
                    n_steps += vals[0].shape[0]
                if self.del_noise and self.clean_noise_validation:
                    if self.need_inputs_for_generating_noise:
                        self.del_noise(*vals)
                    else:
                        self.del_noise()
                inps = list(vals)
                _rvals = self.validate_step(*inps)
                _cost += _rvals

        n_steps = numpy.log(2.)*n_steps
        cost = cost / n_steps

        entropy = cost# (numpy.log(2.))
        ppl = 10**(numpy.log(2)*cost/numpy.log(10))
        return [('cost',entropy), ('ppl',ppl)]

    def get_samples(self, length = 30, temp=1, *inps):
        # Since we have replaced paths for language representations we
        # should no longer need this kind of lazy initialization.

        # if not hasattr(self, 'word_indxs'):
        #     self.load_dict()
        self._get_samples(self, length, temp, *inps)

    def perturb(self, *args, **kwargs):
        if args:
            inps = args
            assert not kwargs
        if kwargs:
            inps = kwargs
            assert not args

        if self.noise_fn:
            if self.clean_before and self.del_noise:
                if self.need_inputs_for_generating_noise:
                    self.del_noise(*args, **kwargs)
                else:
                    self.del_noise()
            inps = self.noise_fn(*args, **kwargs)
        if self.add_noise:
            if self.need_inputs_for_generating_noise:
                self.add_noise(*args, **kwargs)
            else:
                self.add_noise()
        return inps


