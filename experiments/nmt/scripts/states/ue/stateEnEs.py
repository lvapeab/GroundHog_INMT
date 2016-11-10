dict(
#Data
source =            ['/media/HDD_2TB/DATASETS/ue/NMT_DATA/en_30000.binarized_text.h5'],  # Source sequences
target =            ['/media/HDD_2TB/DATASETS/ue/NMT_DATA/es_30000.binarized_text.h5'],  # Target sequences
indx_word =          '/media/HDD_2TB/DATASETS/ue/NMT_DATA/en_30000.ivocab.pkl',        # index -> word dict for the source language
indx_word_target =   '/media/HDD_2TB/DATASETS/ue/NMT_DATA/es_30000.ivocab.pkl', # index -> word dict for the target language
word_indx =          '/media/HDD_2TB/DATASETS/ue/NMT_DATA/en_30000.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt =     '/media/HDD_2TB/DATASETS/ue/NMT_DATA/es_30000.vocab.pkl',    # word -> index dict for the target language

null_sym_source = 30000,
null_sym_target = 30000,

n_sym_source =  30000 + 1,
n_sym_target =  30000 + 1,

dim = 500,
rank_n_approx = 420,
encoder_stack = 1,
decoder_stack = 1,
deep_attention= False,
deep_attention_n_hids = [600, 600],

prefix = '/media/HDD_2TB/MODELS/ue/enes/models/ue_420_500_bs_10',
reload = False,
overwrite = 0,
dropout = .5,
loopIters=1e11,

seqlen = 60,
sort_k_batches = 5,
bs = 10,

bleu_val_frequency=2000*80,
validation_burn_in=20000*80,
hookFreq = 1000*80,
validFreq = 2000*80,
trainFreq = 500*80,
saveFreq = 60,

validation_set='/media/HDD_2TB/DATASETS/ue/DATA/dev.en',
output_validation_set = True,
validation_set_out = '/media/HDD_2TB/MODELS/ue/enes/tmp/ue_420_500.hyp.es',
validation_set_grndtruth='/media/HDD_2TB/DATASETS/ue/DATA/dev.es',

#Unk Replace
unkReplace=True,
mapping = '/media/HDD_2TB/DATASETS/ue/NMT_DATA/topn_es.pkl',
heuristic=1,

# Early stop
patience = 15,
early_stop_time = 24 # In hours
)
