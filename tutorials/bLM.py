import logging
import argparse
import cPickle
from groundhog.datasets import LMIterator
from groundhog.trainer.SGD_momentum import SGD as SGD_m
from groundhog.trainer.SGD import SGD
from groundhog.mainLoop import MainLoop
from groundhog.layers import MultiLayer, \
    RecurrentMultiLayer, \
    RecurrentMultiLayerInp, \
    RecurrentMultiLayerShortPath, \
    RecurrentMultiLayerShortPathInp, \
    RecurrentMultiLayerShortPathInpAll, \
    SoftmaxLayer,\
    HierarchicalSoftmaxLayer, \
    LastState, \
    UnaryOp, \
    DropOp, \
    Operator, \
    Shift, \
    GaussianNoise, \
    SigmoidLayer, \
    MaxPooling, \
    Concatenate
from groundhog.layers import maxpool, \
    maxpool_ntimes, \
    last, \
    last_ntimes, \
    tanh, \
    sigmoid, \
    rectifier, \
    hard_sigmoid, \
    hard_tanh
from groundhog.models import BLM_Model
from theano.scan_module import scan
import numpy
import theano
import pprint
import theano.tensor as TT

linear = lambda x: x
rect = lambda x: TT.maximum(0., x)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("changes", nargs="*", help="Changes to state", default="")
    return parser.parse_args()


def get_state():

    state = {}
    state['seqlen'] = 80
    state['path'] = "/home/lvapeab/smt/software/GroundHog/tutorials/DATA/ue/ue.npz"
    state['dictionary'] = "/home/lvapeab/smt/software/GroundHog/tutorials/DATA/ue/ue_dict.npz"
    state['n_in'] = 10001
    state['n_out'] = 10001
    state['chunks'] = 'words'
    state['seed'] = 123
    state['debug'] = True
    state['on_nan'] = 'warn'

    # DATA
    state['reset'] = -1
    state['minlr'] = float(5e-7)

    # Layers
    # Input weights are sampled from a gaussian with std=scale; this is the
    # standard way to initialize

    state['rank_n_approx'] = 0
    state['inp_nhids'] = '[400]'
    state['inp_activ'] = '[linear]'
    state['inp_bias'] = '[0.]'
    state['inp_sparse'] = -1  # dense
    state['inp_scale'] = .1

    # This is for the output weights
    state['out_scale'] = .1
    state['out_bias_scale'] = -.5
    state['out_sparse'] = -1

    state['dout_nhid'] = '400'
    state['dout_activ'] = '"T.nnet.sigmoid"'
    state['dout_sparse'] = 20
    state['dout_scale'] = 1.
    state['dout_bias'] = '[0]'
    state['dout_init'] = "'sample_weights'"
    state['dout_rank_n_approx'] = 0
    state['dropout'] = 1.

    # HidLayer
    # Hidden units on for the internal layers of DT-RNN. Having a single
    # value results in a standard RNN

    state['nhids'] = '[80]'
    # Activation of each layer
    state['rec_activ'] = '"T.nnet.sigmoid"'
    state['rec_bias'] = '.0'
    state['rec_sparse'] = '20'
    state['rec_scale'] = '1.'

    # sample_weights - you rescale the weights such that the largest
    # singular value is scale
    # sample_weights_classic : just sample weights from a gaussian with std
    # equal to scale
    state['rec_init'] = "'sample_weights'"
    state['rec_layer'] = 'RecurrentMultiLayerShortPathInpAll'

    # SGD params
    state['bs'] = 1  # the size of the minibatch
    state['lr'] = 1.  # initial learning rate
    state['cutoff'] = 1.  # threshold for gradient rescaling
    state['moment'] = 0.995  # -.1 # momentum

    # Do not optimize these
    state['weight_noise'] = True  # white Gaussian noise in weights
    state['weight_noise_amount'] = 0.075  # standard deviation

    # maximal number of updates
    state['loopIters'] = int(1e10)
    # maximal number of minutes to wait until killing job
    state['timeStop'] = 144 * 60  # 144 hours

    # Construct linear connections from input to output. These are factored
    # (like the rank_n) to deal with the possible high dimensionality of the
    # input, but it is a linear projection that feeds into the softmax
    state['shortcut_inpout'] = False
    state['shortcut_rank'] = 200

    state['inpout_nhids'] = '[200]'
    state['inpout_activ'] = '"T.nnet.sigmoid"'
    state['inpout_scale'] = '1.0'
    state['inpout_sparse'] = '20'
    state['inpout_learn_bias'] = '[0]'
    state['inpout_bias'] = '[0]'

    # Main Loop
    # Make this to be a decently large value. Otherwise you waste a lot of
    # memory keeping track of the training error (and other things) at each
    # step + the stdout becomes extremely large
    state['trainFreq'] = 500
    state['hookFreq'] = 4000
    state['validFreq'] = 1000

    state['saveFreq'] = 30  # save every 15 minutes
    state['reload'] = False  # reload
    state['overwrite'] = 1

    # Threshold should be 1.004 for PPL, for entropy (which is what
    # everything returns, it should be much smaller. Running value is 1.0002
    # We should not hyperoptimize this
    state['divide_lr'] = 2.
    state['cost_threshold'] = 1.0002
    state['patience'] = 1
    state['validate_postprocess'] = 'lambda x:10**(x/numpy.log(10))'

    state['truncate_gradient'] = 80  # truncated BPTT
    state['lr_adapt'] = 0  # 1/(1 + n/n0) scheduling
    state['lr_beta'] = 10 * 1900.
    state['lr_start'] = 'on_error'

    state['no_noise_bias'] = True  # do not use weight noise for biases
    state['carry_h0'] = True  # carry over h0 across updates

    state['sample_steps'] = 80

    # Do not change these
    state['minerr'] = -1
    state['shift'] = 1  # n-step forward prediction
    state['cutoff_rescale_length'] = False
    state['nce'] = False

    state['join'] = 'concat',
    state['ntimes'] = False,  # nTimes for the maxpooling

    state['prefix'] = '/home/lvapeab/smt/software/GroundHog/tutorials/models/ue/en_20_60'  # prefix of the save files
    return state


def get_text_data(state):

    def out_format(x, y, r):
        return {'x': x, 'y': y, 'reset': r}

    def out_format_valid(x, y, r):
        return {'x': x, 'y': y, 'reset': r}

    def out_format_test(x, y, r):
        return {'x': x, 'y': y, 'reset': r}

    train_data = LMIterator(
        batch_size=state['bs'],
        path=state['path'],
        stop=-1,
        seq_len=state['seqlen'],
        mode="train",
        chunks=state['chunks'],
        shift=state['shift'],
        output_format=out_format,
        can_fit=True
    )

    valid_data = LMIterator(
        batch_size=state['bs'],
        path=state['path'],
        stop=-1,
        use_infinite_loop=False,
        allow_short_sequences=True,
        seq_len=state['seqlen'],
        mode="valid",
        reset=state['reset'],
        chunks=state['chunks'],
        shift=state['shift'],
        output_format=out_format_valid,
        can_fit=True
    )

    test_data = LMIterator(
        batch_size=state['bs'],
        path=state['path'],
        stop=-1,
        use_infinite_loop=False,
        allow_short_sequences=True,
        seq_len=state['seqlen'],
        mode="test",
        chunks=state['chunks'],
        shift=state['shift'],
        output_format=out_format_test,
        can_fit=True
    )

    if 'wiki' in state['path']:
        test_data = None

    return train_data, valid_data, test_data


class RNN(object):

    def __init__(self, state, rng, reset, prefix='RNN'):
        self.state = state
        self.rng = rng
        self.prefix = prefix
        self.h0 = theano.shared(numpy.zeros((eval(state['nhids'])[-1],), dtype='float32'))
        self.reset = reset
        self.x_emb = None
        self.rec_layer = None
        logger.debug("Layer:" + self.prefix)

    def _create_embedding_layer(self):
        self.emb_words = MultiLayer(
            self.rng,
            n_in=state['n_in'],
            n_hids=eval(state['inp_nhids']),
            activation=eval(state['inp_activ']),
            init_fn='sample_weights_classic',
            weight_noise=state['weight_noise'],
            rank_n_approx=state['rank_n_approx'],
            scale=state['inp_scale'],
            sparsity=state['inp_sparse'],
            learn_bias=True,
            bias_scale=eval(state['inp_bias']),
            name='emb_words')
        logger.debug("_create_embedding_layers")

    def _create_recurrent_layer(self):
        self.rec = eval(state['rec_layer'])(
            self.rng,
            eval(state['nhids']),
            activation=eval(state['rec_activ']),
            bias_scale=eval(state['rec_bias']),
            scale=eval(state['rec_scale']),
            sparsity=eval(state['rec_sparse']),
            init_fn=eval(state['rec_init']),
            weight_noise=state['weight_noise'],
            name='rec')

        logger.debug("_create_recurrent_layer")

    def _create_emb_out_layer(self):
        self.emb_words_out = MultiLayer(
            self.rng,
            n_in=state['n_in'],
            n_hids=eval(state['dout_nhid']),
            activation=linear,
            init_fn='sample_weights_classic',
            weight_noise=state['weight_noise'],
            scale=state['dout_scale'],
            sparsity=state['dout_sparse'],
            rank_n_approx=state['dout_rank_n_approx'],
            learn_bias=False,
            bias_scale=eval(state['dout_bias']),
            name='emb_words_out')

        logger.debug("_create_emb_out_layers")

    def _create_emb_state_layer(self):
        self.emb_state = MultiLayer(
            self.rng,
            n_in=eval(state['nhids'])[-1],
            n_hids=eval(state['dout_nhid']),
            activation=linear,
            init_fn=eval(state['dout_init']),
            weight_noise=state['weight_noise'],
            scale=state['dout_scale'],
            sparsity=state['dout_sparse'],
            learn_bias=True,
            bias_scale=eval(state['dout_bias']),
            name='emb_state')

        logger.debug("_create_emb_state_layer")

    def _create_shortcut_layer(self):
        self.shortcut = MultiLayer(
            self.rng,
            n_in=state['n_in'],
            n_hids=eval(state['inpout_nhids']),
            activation=eval(state['inpout_activ']),
            init_fn='sample_weights_classic',
            weight_noise=state['weight_noise'],
            scale=eval(state['inpout_scale']),
            sparsity=eval(state['inpout_sparse']),
            learn_bias=eval(state['inpout_learn_bias']),
            bias_scale=eval(state['inpout_bias']),
            name='shortcut')

        logger.debug("_create_shortcut_layer")

    def create_layers(self):
        self._create_embedding_layer()
        self._create_recurrent_layer()
        self._create_emb_out_layer()
        self._create_emb_state_layer()
        if state['shortcut_inpout']:
            self._create_shortcut_layer()

    def build_rnn(self, x,
                  no_noise_bias,
                  truncate_gradient):
        self.x_emb = self.emb_words(x, no_noise_bias=no_noise_bias)
        self.rec_layer = self.rec(self.x_emb, n_steps=x.shape[0],
                                  init_state=self.h0 * self.reset,
                                  no_noise_bias=no_noise_bias,
                                  truncate_gradient=truncate_gradient,
                                  batch_size=1)
        logger.debug("Create rec_layer")
        return self


def jobman(state, channel):

    rng = numpy.random.RandomState(state['seed'])

    # load dataset
    train_data, valid_data, test_data = get_text_data(state)

    # Show model info
    logger.debug("_prefix: " + state['prefix'])
    logger.debug("n_in: " + str(state['n_in']))
    logger.debug("n_out: " + str(state['n_out']))
    logger.debug("_seqLength: " + str(state['seqlen']))
    logger.debug("_input units: " + str(state['inp_nhids']))
    logger.debug("_dout units: " + str(state['dout_nhid']))
    logger.debug("_hidden units: " + str(state['nhids']))
    logger.debug("_Shortcut inpout: " + str(state['shortcut_inpout']))
    logger.debug("_Layer combination: " + str(state['join']))
    logger.debug("_NCE: " + str(state['nce']))
    if state['shortcut_inpout']:
        logger.debug("_shortcut rank: " + str(state['shortcut_rank']))

    # Define Theano Input Variables
    x = TT.lvector('x')
    y = TT.lvector('y')
    reset = TT.scalar('reset')
    h0 = theano.shared(numpy.zeros((eval(state['nhids'])[-1],), dtype='float32'))

    # Neural Implementation of the Operators: \oplus
    logger.debug("_create forward RNN")
    forward = RNN(state, rng, reset, prefix="forward_rnn")
    forward.create_layers()
    forward_training = forward.build_rnn(x,
                                         no_noise_bias=state['no_noise_bias'],
                                         truncate_gradient=state['truncate_gradient'])

    logger.debug("_create backward RNN")

    backward = RNN(state, rng, reset, prefix="backward_rnn")
    backward.create_layers()
    backward_training = backward.build_rnn(x[::-1],
                                           no_noise_bias=state['no_noise_bias'],
                                           truncate_gradient=state['truncate_gradient'])
    # Reverse time for backward representations.
    backward_training.rec_layer.out = backward_training.rec_layer.out[::-1]

    # Hidden State: Combine emb_state and emb_words_out
    # 1.  Define an activation layer
    outhid_activ = UnaryOp(activation=eval(state['dout_activ']))

    # 2. Define a dropout layer
    outhid_dropout = DropOp(dropout=state['dropout'], rng=rng)

    logger.debug("Create output_layer")

    # Softmax Layer
    if state['join'] == 'concat':
        output_layer = SoftmaxLayer(
            rng,
            eval(state['dout_nhid']+'*2'),
            state['n_out'],
            scale=state['out_scale'],
            bias_scale=state['out_bias_scale'],
            init_fn="sample_weights_classic",
            weight_noise=state['weight_noise'],
            sparsity=state['out_sparse'],
            sum_over_time=True,
            use_nce=state['nce'],
            name='out')

    else:
        output_layer = SoftmaxLayer(
            rng,
            eval(state['dout_nhid']),
            state['n_out'],
            scale=state['out_scale'],
            bias_scale=state['out_bias_scale'],
            init_fn="sample_weights_classic",
            weight_noise=state['weight_noise'],
            sparsity=state['out_sparse'],
            sum_over_time=True,
            use_nce=state['nce'],
            name='out')
    # Learning rate scheduling (1/(1+n/beta))
    state['clr'] = state['lr']

    def update_lr(obj, cost):
        stp = obj.step
        if isinstance(obj.state['lr_start'], int) and stp > obj.state['lr_start']:
            time = float(stp - obj.state['lr_start'])
            new_lr = obj.state['clr'] / (1 + time / obj.state['lr_beta'])
            obj.lr = new_lr

    if state['lr_adapt']:
        forward_training.add_schedule(update_lr)
        backward_training.add_schedule(update_lr)

    if state['shortcut_inpout']:
        additional_inputs_f = [forward_training.rec_layer, forward_training.shortcut(x)]
        additional_inputs_b = [backward_training.rec_layer, backward_training.shortcut(x[::-1])]
        additional_inputs = [additional_inputs_f + additional_inputs_b]
    else:
        additional_inputs = [forward_training.rec_layer + backward_training.rec_layer]

    # Training model
    # Neural Implementations of the Language Model

    # Output intermediate layer
    logger.debug("_build train model")
    training_components = []
    
    if state['join'] == 'concat':
        training_components.append(forward_training.rec_layer + forward_training.emb_words_out(x))
        training_components.append(backward_training.rec_layer + backward_training.emb_words_out(x[::-1]))
        outhid = Concatenate(axis=1)(*training_components)
        outhid = outhid_activ(outhid)
        outhid = outhid_dropout(outhid)
    else:
        outhid = outhid_activ(forward_training.rec_layer + forward_training.emb_words_out(x) +
                              backward_training.rec_layer + backward_training.emb_words_out(x[::-1]))
        outhid = outhid_dropout(outhid)

    train_model = output_layer(outhid,
                               no_noise_bias=state['no_noise_bias'],
                               additional_inputs=additional_inputs).train(target=y,
                                                                          scale=numpy.float32(1. / state['seqlen']))
    nw_h0_f = forward_training.rec_layer.out[forward_training.rec_layer.out.shape[0] - 1]
    nw_h0_b = backward_training.rec_layer.out[backward_training.rec_layer.out.shape[0] - 1]

    if state['carry_h0']:
        train_model.updates += [(h0, nw_h0_f, nw_h0_b)]

    # Validation model
    logger.debug("_build validation model")
    logger.debug("_create_forward_RNN")
    forward_valid = forward.build_rnn(x,
                                      no_noise_bias=state['no_noise_bias'],
                                      truncate_gradient=state['truncate_gradient'])

    logger.debug("_create_forward_RNN")
    backward_valid = backward.build_rnn(x[::-1],
                                        no_noise_bias=state['no_noise_bias'],
                                        truncate_gradient=state['truncate_gradient'])
    backward_valid.rec_layer.out = backward_valid.rec_layer.out[::-1]

    h0val = theano.shared(numpy.zeros((eval(state['nhids'])[-1],), dtype='float32'))

    nw_h0_val_f = forward_valid.rec_layer.out[forward_valid.rec_layer.out.shape[0] - 1]
    nw_h0_val_b = backward_valid.rec_layer.out[backward_valid.rec_layer.out.shape[0] - 1]
    nw_h0 = nw_h0_val_f + nw_h0_val_b

    valid_components = []

    if state['join'] == 'concat':
        valid_components.append(forward_valid.rec_layer + forward_valid.emb_words_out(x))
        valid_components.append(backward_valid.rec_layer + backward_valid.emb_words_out(x[::-1]))
        outhid = Concatenate(axis=1)(*valid_components)
        outhid = outhid_activ(outhid)
        outhid = outhid_dropout(outhid)
    else:
        outhid = outhid_activ(forward_valid.rec_layer + forward_valid.emb_words_out(x) +
                              backward_valid.rec_layer + backward_valid.emb_words_out(x[::-1]))
        outhid = outhid_dropout(outhid)

    valid_model = output_layer(outhid,
                               additional_inputs=additional_inputs,
                               use_noise=False).validate(target=y, sum_over_time=True)

    valid_updates = []

    if state['carry_h0']:
        valid_updates = [(h0val, nw_h0)]

    valid_fn = theano.function([x, y, reset], valid_model.cost,
                               name='valid_fn', updates=valid_updates,
                               on_unused_input='warn'
                               )

    # Build and Train a Model
    # Define a model
    model = BLM_Model(
        cost_layer=train_model,
        weight_noise_amount=state['weight_noise_amount'],
        valid_fn=valid_fn,
        clean_before_noise_fn=False,
        noise_fn=None,
        rng=rng
    )

    if state['reload']:
        model.load(state['prefix'] + 'model.npz')

    # Define a trainer
    # Training algorithm (SGD)

    if state['moment'] < 0:
        algo = SGD(model, state, train_data)

    else:
        algo = SGD_m(model, state, train_data)

    # Main loop of the trainer
    main = MainLoop(train_data,
                    valid_data,
                    test_data,
                    model,
                    algo,
                    state,
                    channel,
                    train_cost=False,
                    hooks=None,
                    validate_postprocess=eval(state['validate_postprocess']))
    # Run!
    main.main()


if __name__ == '__main__':

    state = get_state()
    args = parse_args()
    if args.state:
        logger.debug('_loading state: ' + args.state)
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    jobman(state, None)
