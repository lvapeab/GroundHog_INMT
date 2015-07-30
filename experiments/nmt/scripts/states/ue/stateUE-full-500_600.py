dict(
#Datos de entrenamiento
source = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/es.binarized_text.h5'],  # Source sequences
target = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/en.binarized_text.h5'],  # Target sequences    
indx_word = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/es.ivocab.pkl',        # index -> word dict for the source language
indx_word_target = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/en.ivocab.pkl', # index -> word dict for the target language
word_indx = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/es.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/en.vocab.pkl',    # word -> index dict for the target language
     
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
dim = 600,
rank_n_approx = 500,
maxout_part = 2.,
prefix = '/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_11k_full_600_500_',
reload = True,
overwrite = 1,
dropout = 1.0,
source_encoding = 'utf8',
target_encoding = 'utf8',
seqlen = 50,
sort_k_batches = 20,
bs = 80 ,
deep_attention= True,
deep_attention_n_hids = [600, 600],
deep_attention_acts= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) '],
bleu_val_frequency=2000,
validation_burn_in=30000,
hookFreq = 8000,
validFreq = 3000,
trainFreq = 1000,
validation_set='/home/lvapeab/smt/tasks/ue/esen/DATA/test.es',
output_validation_set = True,
validation_set_out = '/home/lvapeab/smt/tasks/ue/esen/NMT/tmp/ue.test.500-620.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/ue/esen/DATA/test.en',
saveFreq = 30
)
