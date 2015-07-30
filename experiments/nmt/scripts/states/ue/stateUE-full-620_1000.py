dict(
#Data
source = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/es.binarized_text.h5'],  # Source sequences
target = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/en.binarized_text.h5'],  # Target sequences
indx_word = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/es.ivocab.pkl',        # index -> word dict for the source language
indx_word_target = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/en.ivocab.pkl', # index -> word dict for the target language
word_indx = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/es.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/en.vocab.pkl',    # word -> index dict for the target language

source_encoding = 'utf8',
target_encoding = 'utf8',
encoding = 'utf8',

oov = 'UNK',
unk_sym_source = 1,
unk_sym_target = 1,
null_sym_source = 11000,
null_sym_target = 11000,
n_sym_source =  11000 + 1,
n_sym_target =  11000 + 1,

last_forward = False,
last_backward = False,
forward = True,
backward = True,                                                            
search = True,
bigram = True, # shortcut from the previous word to the current one
dec_rec_layer= 'RecurrentLayerWithSearch',
dim = 1000,
rank_n_approx = 620,
maxout_part = 2.,

prefix = '/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_full_620_1000_',
reload = True,
overwrite = 1,
dropout = 1.,
seqlen = 55,
sort_k_batches = 20,
bs = 80,
loopIters=1e12,


deep_attention= True,
deep_attention_n_hids = [1000, 1000],
deep_attention_acts= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) '],
take_top = False,

bleu_val_frequency=2000,
validation_burn_in=10000,
hookFreq = 8000,
validFreq = 2000,
trainFreq = 1000,
validation_set='/home/lvapeab/smt/tasks/ue/esen/DATA/test.es',
output_validation_set = True,
validation_set_out = '/home/lvapeab/smt/tasks/ue/esen/NMT/tmp/ue.test.620-700.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/ue/esen/DATA/test.en',
saveFreq = 30,

# Early Stopping Stuff
patience = 1,
lr = 1e-3,
minlr = 0,



# Random weight noise regularization settings
weight_noise = True,
weight_noise_rec = False,
weight_noise_amount = 0.01,

# Threshold to clip the gradient
cutoff = 1.,
# A magic gradient clipping option that you should never change...
cutoff_rescale_length = 0.,
# Choose optimization algorithm
algo = 'SGD_adadelta',

# Adadelta hyperparameters
adarho = 0.95,
adaeps = 1e-6,
# Early stopping configuration
# WARNING: was never changed during machine translation experiments,
# as early stopping was not used.

# Turns on trimming the trailing paddings from batches
# consisting of short sentences.
trim_batches = True,

# Loop through the data
use_infinite_loop = True,
# Start from a random entry
shuffle = False



)
