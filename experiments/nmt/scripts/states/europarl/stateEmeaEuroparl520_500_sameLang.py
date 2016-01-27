dict(
#Data
source =          ['/home/lvapeab/smt/tasks/emea_europarl/NMT/DATA/en.binarized_text.h5'],  # Source sequences
target =          ['/home/lvapeab/smt/tasks/emea_europarl/NMT/DATA/en.binarized_text.h5'],  # Target sequences
indx_word =        '/home/lvapeab/smt/tasks/emea_europarl/NMT/DATA/en.ivocab.pkl',        # index -> word dict for the source language
indx_word_target = '/home/lvapeab/smt/tasks/emea_europarl/NMT/DATA/en.ivocab.pkl', # index -> word dict for the target language
word_indx =        '/home/lvapeab/smt/tasks/emea_europarl/NMT/DATA/en.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt =   '/home/lvapeab/smt/tasks/emea_europarl/NMT/DATA/en.vocab.pkl',    # word -> index dict for the target language

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
dim = 500,
rank_n_approx = 520,
maxout_part = 2.,

prefix = '/home/lvapeab/smt/tasks/emea_europarl/NMT/models/emea_europarl_15k_520_500_en.en_',
reload = True,
overwrite = 1,
dropout = .5,
seqlen = 50,
sort_k_batches = 20,
bs = 80,
loopIters=1e10,

deep_attention= True,
deep_attention_n_hids = [500, 500],
deep_attention_acts= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) '],

bleu_val_frequency=6000,
validation_burn_in=20000,
hookFreq = 3000,
validFreq = 6000,
trainFreq = 1000,
validation_set='/home/lvapeab/smt/tasks/emea_europarl/DATA/khresmoi-summary-dev.lowercased.en',
output_validation_set = True,
validation_set_out = '/home/lvapeab/smt/tasks/emea_europarl/NMT/tmp/dev.hyp',
validation_set_grndtruth='/home/lvapeab/smt/tasks/emea_europarl/DATA/khresmoi-summary-dev.lowercased.en',
saveFreq = 119,

# Early Stopping Stuff
patience = 1,
lr = 1e-3,
minlr = 0
)
