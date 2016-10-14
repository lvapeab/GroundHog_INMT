#!/usr/bin/env python

import cPickle
import logging

import numpy

from experiments.nmt import \
    RNNEncoderDecoder, \
    prototype_phrase_state
# from experiments.nmt.PE_isles.isles_utils import compute_mouse_movements, find_isles, is_sublist, subfinder
from nmt.online.online_utils import loadSourceLanguageFromState, loadTargetLanguageFromState
from groundhog.datasets.UnbufferedDataIterator import UnbufferedDataIterator
from groundhog.trainer.SGD import SGD

logger = logging.getLogger(__name__)


from experiments.nmt.PE_isles.postediting_sampling_isles_ensemble_online import BeamSearch

state_f = "/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_state.pkl"
model_f ="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu52.npz"
source = '/home/lvapeab/smt/tasks/xerox/DATA/dev.es'
target = '/home/lvapeab/smt/tasks/xerox/DATA/dev.en'

state = prototype_phrase_state()
with open(state_f) as src:
    state.update(cPickle.load(src))
rng = numpy.random.RandomState(state['seed'])
num_models = 1
num_sentences = 1
logger.info("Using an ensemble of %d models" % num_models)
enc_decs = []
lm_models = []
alignment_fns = []
algos = []
sourceLanguage = loadSourceLanguageFromState(state)
targetLanguage = loadTargetLanguageFromState(state)
batch_iter = UnbufferedDataIterator(source, target, state, sourceLanguage.word_indx,
                                    targetLanguage.word_indx, sourceLanguage.indx_word,
                                    targetLanguage.indx_word, num_sentences, state['seqlen'], None)

for i in xrange(num_models):
    enc_decs.append(RNNEncoderDecoder(state, rng, skip_init=True,
                                      compute_alignment=False))
    enc_decs[i].build()
    lm_models.append(enc_decs[i].create_lm_model())
    lm_models[i].load(model_f)

idict_src = sourceLanguage.indx_word  # cPickle.load(open(state['indx_word'], 'r'))
indx_word = cPickle.load(open(state['word_indx'], 'rb'))
unk_id = state['unk_sym_target']
word2index = targetLanguage.word_indx  # cPickle.load(open(state['word_indx_trgt'], 'r'))
indx2word_trg = targetLanguage.indx_word  # cPickle.load(open(state['indx_word_target'], 'rb'))
eos_id = state['null_sym_target']

sampler = None
logger.info('Creating beam search')
beam_search = BeamSearch(enc_decs)
beam_search.compile()


for i in xrange(num_models):
        print "lm_models[", i, "]=", lm_models[i]
        algos.append(SGD(lm_models[i], state, batch_iter))
        print "algos[", i, "]=", algos[i]
