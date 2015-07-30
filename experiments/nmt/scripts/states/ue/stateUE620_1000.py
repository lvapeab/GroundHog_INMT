dict(
source = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/training.es.binarized_text.h5'],
target = ['/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/training.en.binarized_text.h5'],
indx_word = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/training.es.ivocab.pkl',
indx_word_target = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/training.en.ivocab.pkl',
word_indx = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/training.es.vocab.pkl',
word_indx_trgt = '/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/training.en.vocab.pkl',        
oov = 'UNK',
unk_sym_source = 1,
unk_sym_target = 1,
null_sym_source = 10000,
null_sym_target = 10000,
n_sym_source =  10000 + 1,
n_sym_target =  10000  + 1,

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
prefix = '/home/lvapeab/nn_models/msc/ue_full_620_1000_validBLEU',
reload = True,
overwrite = 1,
source_encoding = 'utf8',
target_encoding = 'utf8',
seqlen = 50,
sort_k_batches = 20,
bs = 80 ,
deep_attention= True,
deep_attention_n_hids = [1000, 1000],
deep_attention_acts= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) '],
bleu_val_frequency=500,
validation_burn_in=500,
hookFreq = 2000,
validFreq = 1000,
trainFreq = 500,                                                                    
validation_set='/home/lvapeab/smt/tasks/ue/esen/DATA/dev.es',                                                                                                                                                 
output_validation_set = True,                                                                                                                                                         
validation_set_out = '/home/lvapeab/ue.dev.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/ue/esen/DATA/dev.en'
)
