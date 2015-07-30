dict(
source = ['/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/full_vocab/en.binarized_text.shuffled.h5'],
target = ['/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/full_vocab/es.binarized_text.shuffled.h5'],
indx_word = '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/full_vocab/en.ivocab.pkl',
indx_word_target = '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/full_vocab/es.ivocab.pkl',
word_indx = '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/full_vocab/en.vocab.pkl',
word_indx_trgt = '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/full_vocab/es.vocab.pkl',        
oov = 'UNK',
unk_sym_source = 1,
unk_sym_target = 1,
null_sym_source = 11424,
null_sym_target = 14479,
n_sym_source =  11424 + 1,
n_sym_target =  14479 + 1,

last_forward = False,
last_backward = False,
forward = True,
backward = True,                                                            
search = True,
bigram = True, # shortcut from the previous word to the current one
dec_rec_layer= 'RecurrentLayerWithSearch',
dim = 500,
rank_n_approx = 620,
maxout_part = 2.,
prefix = '/home/lvapeab/smt/tasks/xerox/enes/NMT/xerox_fullVocab_400_620_',
reload = True,
overwrite = 1,
dropout = 1.0,
source_encoding = 'utf8',
target_encoding = 'utf8',
seqlen = 50,
sort_k_batches = 20,
bs = 80 ,
deep_attention= True,
deep_attention_n_hids = [500, 500],
deep_attention_acts= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) '],
bleu_val_frequency=2000,
validation_burn_in=1000,
hookFreq = 2000,
validFreq = 2000,
trainFreq = 500,
validation_set='/home/lvapeab/smt/tasks/xerox/enes/DATA/test.en',
output_validation_set = True,
validation_set_out = 'home/lvapeab/smt/tasks/xerox/enes/NMT/tmp/xerox.test.400-620.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/xerox/enes/DATA/test.es'
)
