dict(
source=         ['/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/es.binarized_text.shuffled.h5'],
target=         ['/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/en.binarized_text.shuffled.h5'],
indx_word=       '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/es.ivocab.pkl',
indx_word_target='/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/en.ivocab.pkl',
word_indx=       '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/es.vocab.pkl',
word_indx_trgt=  '/home/lvapeab/smt/tasks/xerox/enes/NMT/DATA/en.vocab.pkl',

null_sym_source=2, #16790,
null_sym_target=2,#14060,
n_sym_source=16790 + 1,
n_sym_target=14060 + 1,

dim=500,
rank_n_approx=300,
encoder_stack = 2,
decoder_stack = 1,
deep_attention=False,
deep_attention_n_hids=[500, 500],
prefix='/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_300_500_2111_',
reload=True,
overwrite=1,
dropout=0.5,
loopIters=1e10,
bleu_val_frequency=2000,
validation_burn_in=20000,
hookFreq=1000,
validFreq=1000,
trainFreq=500,
saveFreq=60,

validation_set='/home/lvapeab/smt/tasks/xerox/DATA/dev.es',
validation_set_out='/home/lvapeab/smt/tasks/xerox/esen/NMT/tmp/xerox_400_400_hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/xerox/DATA/dev.en',

# Early stop
patience=30,
early_stop_time=24, # In hours
)
