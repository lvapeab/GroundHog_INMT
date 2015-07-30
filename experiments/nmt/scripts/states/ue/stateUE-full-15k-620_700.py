dict(
#Data
source = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/15k/es.binarized_text.h5'],  # Source sequences
target = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/15k/en.binarized_text.h5'],  # Target sequences
indx_word = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/15k/es.ivocab.pkl',        # index -> word dict for the source language
indx_word_target = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/15k/en.ivocab.pkl', # index -> word dict for the target language
word_indx = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/15k/es.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/15k/en.vocab.pkl',    # word -> index dict for the target language

source_encoding = 'utf8',
target_encoding = 'utf8',
encoding = 'utf8',

oov = 'UNK',
unk_sym_source = 1,
unk_sym_target = 1,
null_sym_source = 15000,
null_sym_target = 15000,
n_sym_source =  15000 + 1,
n_sym_target =  15000 + 1,

last_forward = False,
last_backward = False,
forward = True,
backward = True,                                                            
search = True,
bigram = True, # shortcut from the previous word to the current one
dec_rec_layer= 'RecurrentLayerWithSearch',
dim = 700,
rank_n_approx = 620,
maxout_part = 2.,

prefix = '/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_11k_full_620_700_',
reload = True,
overwrite = 1,
dropout = 1.,
seqlen = 60,
sort_k_batches = 20,
bs = 80,
loopIters=1e9,


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
minlr = 0


)
