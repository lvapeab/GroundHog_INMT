dict(
source=         ['/home/lvapeab/smt/tasks/xerox/esen/NMT/DATA/lowercased/es.binarized_text.shuffled.h5'],
target=         ['/home/lvapeab/smt/tasks/xerox/esen/NMT/DATA/lowercased/en.binarized_text.shuffled.h5'],
indx_word=       '/home/lvapeab/smt/tasks/xerox/esen/NMT/DATA/lowercased/es.ivocab.pkl',
indx_word_target='/home/lvapeab/smt/tasks/xerox/esen/NMT/DATA/lowercased/en.ivocab.pkl',
word_indx=       '/home/lvapeab/smt/tasks/xerox/esen/NMT/DATA/lowercased/es.vocab.pkl',
word_indx_trgt=  '/home/lvapeab/smt/tasks/xerox/esen/NMT/DATA/lowercased/en.vocab.pkl',

null_sym_source=14479,
null_sym_target=11424,
n_sym_source=14479 + 1,
n_sym_target=11424 + 1,

dim=300,
rank_n_approx=250,
encoder_stack = 2,
decoder_stack = 1,
deep_attention=False,
deep_attention_n_hids=[500, 500],
prefix='/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_low_250_300_',
reload=True,
overwrite=1,
dropout=0.5,
loopIters=1e10,
bleu_val_frequency=2000,
validation_burn_in=50000,
hookFreq=1000,
validFreq=1000,
trainFreq=500,
saveFreq=60,

validation_set='/home/lvapeab/smt/tasks/xerox/DATA/lowercased/dev.es',
validation_set_out='/home/lvapeab/smt/tasks/xerox/esen/NMT/tmp/xerox_250_300_hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/xerox/DATA/lowercased/dev.en',

# Early stop
patience=30,
early_stop_time=24, # In hours
)
