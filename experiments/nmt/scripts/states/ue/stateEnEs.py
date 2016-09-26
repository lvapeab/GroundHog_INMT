dict(
#Data
source =            ['/home/lvapeab/smt/tasks/ue/enes/NMT/DATA/en_30k.binarized_text.h5'],  # Source sequences
target =            ['/home/lvapeab/smt/tasks/ue/enes/NMT/DATA/es_30k.binarized_text.h5'],  # Target sequences
indx_word =          '/home/lvapeab/smt/tasks/ue/enes/NMT/DATA/en_30k.ivocab.pkl',        # index -> word dict for the source language
indx_word_target =   '/home/lvapeab/smt/tasks/ue/enes/NMT/DATA/es_30k.ivocab.pkl', # index -> word dict for the target language
word_indx =          '/home/lvapeab/smt/tasks/ue/enes/NMT/DATA/en_30k.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt =     '/home/lvapeab/smt/tasks/ue/enes/NMT/DATA/es_30k.vocab.pkl',    # word -> index dict for the target language

null_sym_source = 30000,
null_sym_target = 30000,

n_sym_source =  30000 + 1,
n_sym_target =  30000 + 1,

dim = 712,
rank_n_approx = 382,
encoder_stack = 1,
decoder_stack = 1,
deep_attention= False,
deep_attention_n_hids = [600, 600],

prefix = '/home/lvapeab/smt/tasks/ue/enes/NMT/models/ue_712_382_',
reload = False,
overwrite = 0,
dropout = .5,
loopIters=1e11,

seqlen = 50,
sort_k_batches = 20,
bs = 80,

bleu_val_frequency=2000,
validation_burn_in=20000,
hookFreq = 1000,
validFreq = 2000,
trainFreq = 500,
saveFreq = 60,

validation_set='/home/lvapeab/smt/tasks/ue/DATA/test.en',
output_validation_set = True,
validation_set_out = '/home/lvapeab/smt/tasks/ue/enes/NMT/tmp/ue_712_382.hyp.es',
validation_set_grndtruth='/home/lvapeab/smt/tasks/ue/DATA/test.es',

#Unk Replace
unkReplace=True,
mapping = '/home/lvapeab/smt/tasks/ue/enes/NMT/DATA/topn.pkl',
heuristic=1,

# Early stop
patience = 15,
early_stop_time = 24 # In hours
)
