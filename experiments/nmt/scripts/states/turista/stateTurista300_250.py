dict(
#Data
source = ['/home/lvapeab/smt/tasks/turista/NMT/DATA/es.binarized_text.h5'],
target = ['/home/lvapeab/smt/tasks/turista/NMT/DATA/en.binarized_text.h5'],                         
indx_word = '/home/lvapeab/smt/tasks/turista/NMT/DATA/es.ivocab.pkl',
indx_word_target = '/home/lvapeab/smt/tasks/turista/NMT/DATA/en.ivocab.pkl',
word_indx = '/home/lvapeab/smt/tasks/turista/NMT/DATA/es.vocab.pkl',
word_indx_trgt = '/home/lvapeab/smt/tasks/turista/NMT/DATA/en.vocab.pkl',


source_encoding = 'utf8',
target_encoding = 'utf8',
encoding = 'utf8',
oov = 'UNK',



unk_sym_source = 1,
unk_sym_target = 1,
null_sym_source = 686,
null_sym_target = 513,
n_sym_source =  686 + 1,
n_sym_target =  513  + 1,


last_forward = False,
last_backward = False,
forward = True,
backward = True,                                                            
search = True,                                                                                
dec_rec_layer= 'RecurrentLayerWithSearch',
dim = 250,
rank_n_approx = 300,
maxout_part = 2.,



prefix = '/home/lvapeab/smt/tasks/turista/NMT/models/turista_250_300_',
reload = True,
overwrite = 1,
dropout = .5,
seqlen = 30,
sort_k_batches = 20,
bs = 80,
loopIters=1e9,

deep_attention= True,
deep_attention_n_hids= [250,250],
deep_attention_acts= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) '],



bleu_val_frequency=6000,
validation_burn_in=6000,
hookFreq = 12000,
validFreq = 2000,
trainFreq = 1000,
validation_set='/home/lvapeab/smt/tasks/turista/DATA/dev.es',
output_validation_set = True,
validation_set_out = '/home/lvapeab/smt/tasks/turista/NMT/tmp/dev.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/turista/DATA/dev.en',
saveFreq = 30,

# Early Stopping Stuff
patience = 1,
lr = 1e-3,
minlr = 0

)
