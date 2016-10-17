dict(
#Data
source =            ['/home/lvapeab/smt/tasks/europarl/DATA/NMT/fr_30k.binarized_text.shuffled.h5'],  # Source sequences
target =            ['/home/lvapeab/smt/tasks/europarl/DATA/NMT/en_30k.binarized_text.shuffled.h5'],  # Target sequences
indx_word =          '/home/lvapeab/smt/tasks/europarl/DATA/NMT/fr_30k.ivocab.pkl',        # index -> word dict for the source language
indx_word_target =   '/home/lvapeab/smt/tasks/europarl/DATA/NMT/en_30k.ivocab.pkl', # index -> word dict for the target language
word_indx =          '/home/lvapeab/smt/tasks/europarl/DATA/NMT/fr_30k.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt =     '/home/lvapeab/smt/tasks/europarl/DATA/NMT/en_30k.vocab.pkl',    # word -> index dict for the target language

null_sym_source = 30000,
null_sym_target = 30000,

n_sym_source =  30000 + 1,
n_sym_target =  30000 + 1,

dim = 1000,
rank_n_approx = 620,
encoder_stack = 1,
decoder_stack = 1,
deep_attention= False,
deep_attention_n_hids = [600, 600],

prefix = '/home/lvapeab/smt/tasks/europarl/fren/NMT/models/europarl_1000_620_',
reload = True,
overwrite = 0,
dropout = .5,
loopIters=1e11,

seqlen = 50,
sort_k_batches = 20,
bs = 80,

bleu_val_frequency=20000,
validation_burn_in=180000,
hookFreq = 1000,
validFreq = 2000,
trainFreq = 500,
saveFreq = 60,

validation_set='/home/lvapeab/smt/tasks/europarl/DATA/dev.fr',
output_validation_set = True,
validation_set_out = '/home/lvapeab/smt/tasks/europarl/fren/NMT/tmp/europarl_1000_620.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/europarl/DATA/dev.en',

#Unk Replace
unkReplace=True,
mapping = '/home/lvapeab/smt/tasks/europarl/DATA/NMT/topn_en.pkl',
heuristic=1,

# Early stop
patience = 15,
early_stop_time = 24 # In hours
)
