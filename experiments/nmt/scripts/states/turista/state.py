dict(
#Data
source = ['/home/lvapeab/smt/tasks/turista/NMT/DATA/es.binarized_text.h5'],
target = ['/home/lvapeab/smt/tasks/turista/NMT/DATA/en.binarized_text.h5'],                         
indx_word = '/home/lvapeab/smt/tasks/turista/NMT/DATA/es.ivocab.pkl',
indx_word_target = '/home/lvapeab/smt/tasks/turista/NMT/DATA/en.ivocab.pkl',
word_indx = '/home/lvapeab/smt/tasks/turista/NMT/DATA/es.vocab.pkl',
word_indx_trgt = '/home/lvapeab/smt/tasks/turista/NMT/DATA/en.vocab.pkl',


unk_sym_source = 1,
unk_sym_target = 1,
null_sym_source = 686,
null_sym_target = 513,
n_sym_source =  686 + 1,
n_sym_target =  513  + 1,


dim = 20,
rank_n_approx = 30,
encoder_stack = 2,
decoder_stack = 2,
deep_attention= True,
deep_attention_n_hids = [20, 20],

prefix = '/home/lvapeab/smt/tasks/turista/esen/NMT/models/turista_20_30_2221_',
reload = False,
overwrite = 1,
dropout = .5,
loopIters=1e4,

seqlen = 30,
sort_k_batches = 20,
bs = 80,


bleu_val_frequency=200,
validation_burn_in=300,
hookFreq = 200,
validFreq = 200,
trainFreq = 100,
saveFreq = 60,

validation_set='/home/lvapeab/smt/tasks/turista/DATA/dev.es',
output_validation_set = True,
validation_set_out = '/home/lvapeab/smt/tasks/turista/NMT/tmp/dev.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/turista/DATA/dev.en',


# Early stop
patience = 15,
early_stop_time = 24 # In hours
)