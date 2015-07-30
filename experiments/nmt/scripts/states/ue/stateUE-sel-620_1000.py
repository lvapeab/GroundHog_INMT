dict(
#Datos de entrenamiento
source = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/reduced.es.binarized_text.h5'],  # Source sequences
target = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/reduced.en.binarized_text.h5'],  # Target sequences    
indx_word = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/reduced.es.ivocab.pkl',        # index -> word dict for the source language
indx_word_target = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/reduced.en.ivocab.pkl', # index -> word dict for the target language
word_indx = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/reduced.es.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/reduced.en.vocab.pkl',    # word -> index dict for the target language
     
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
dim = 1000,
rank_n_approx = 620,
maxout_part = 2.,
prefix = '/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_15k_1000_620_',
reload = True,
overwrite = 1,
dropout = 1.0,
source_encoding = 'utf8',
target_encoding = 'utf8',
seqlen = 50,
sort_k_batches = 20,
bs = 80 ,
deep_attention= True,
deep_attention_n_hids = [1000, 1000],
deep_attention_acts= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) '],
bleu_val_frequency=3000,
validation_burn_in=50000,
hookFreq =82000,
validFreq = 3000,
trainFreq = 1000,
validation_set='/home/lvapeab/smt/tasks/ue/esen/DATA/test.es',
output_validation_set = True,
validation_set_out = '/home/lvapeab/smt/tasks/ue/esen/NMT/tmp/ue.test.1000-620.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/ue/esen/DATA/test.en'
)
