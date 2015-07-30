dict(
source = ['/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/red/wvalid.en.binarized_text.shuffled.h5'],
target = ['/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/red/wvalid.es.binarized_text.shuffled.h5'],                         
indx_word = '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/red/wvalid.en.ivocab.pkl',
indx_word_target = '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/red/wvalid.es.ivocab.pkl',
word_indx = '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/red/wvalid.en.vocab.pkl',
word_indx_trgt = '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/red/wvalid.es.vocab.pkl',        
oov = 'UNK',
unk_sym_source = 1,
unk_sym_target = 1,
null_sym_source = 3564,
null_sym_target = 3564,
n_sym_source =  3564 + 1,
n_sym_target =  3564 + 1,
last_forward = False,
last_backward = False,
forward = True,
backward = True,                                                            
search = True,                                                                                
dec_rec_layer= 'RecurrentLayerWithSearch',
dim = 600,
rank_n_approx = 1000,
maxout_part = 2.,
prefix = '/home/lvapeab/nn_models/msc/xerox_phrase_rec_sel_wvalid_1000_600_',
reload = True,
overwrite = 1,

#Sampling details                                                                                                                             
hookFreq = 500,
validFreq = 100,
trainFreq = 50
)
