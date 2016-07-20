dict(
source=         ['/home/lvapeab/smt/tasks/emea/enfr/NMT/DATA/en_30k.binarized_text.shuffled.h5'],
target=         ['/home/lvapeab/smt/tasks/emea/enfr/NMT/DATA/fr_30k.binarized_text.shuffled.h5'],
indx_word=       '/home/lvapeab/smt/tasks/emea/enfr/NMT/DATA/en_30k.ivocab.pkl',
indx_word_target='/home/lvapeab/smt/tasks/emea/enfr/NMT/DATA/fr_30k.ivocab.pkl',
word_indx=       '/home/lvapeab/smt/tasks/emea/enfr/NMT/DATA/en_30k.vocab.pkl',
word_indx_trgt=  '/home/lvapeab/smt/tasks/emea/enfr/NMT/DATA/fr_30k.vocab.pkl',

null_sym_source = 30000,
null_sym_target = 30000,

n_sym_source =  30000 + 1,
n_sym_target =  30000 + 1,

dim = 520,
rank_n_approx = 400,
encoder_stack = 2,
decoder_stack = 1,
deep_attention= True,
deep_attention_n_hids = [520, 520],

prefix='/home/lvapeab/smt/tasks/emea/enfr/NMT/models/emea_400_520_',
reload = True,
overwrite = 0,
dropout = .5,
loopIters=1e11,

seqlen = 50,
sort_k_batches = 20,
bs = 80,

bleu_val_frequency=2000,
validation_burn_in=20000,
hookFreq = 1000,
validFreq = 2000,
trainFreq = 500,
saveFreq = 60,


output_validation_set=True,
validation_set='/home/lvapeab/smt/tasks/emea/DATA/test.en',
validation_set_out='/home/lvapeab/smt/tasks/emea/enfr/NMT/tmp/emea_400_520.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/emea/DATA/test.fr',

# Early stop
patience = 15,
early_stop_time = 24 # In hours
)
