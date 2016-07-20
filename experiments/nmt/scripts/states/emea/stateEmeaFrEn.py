dict(
source=         ['/home/lvapeab/smt/tasks/emea/fren/NMT/DATA/fr.binarized_text.shuffled.h5'],
target=         ['/home/lvapeab/smt/tasks/emea/fren/NMT/DATA/en.binarized_text.shuffled.h5'],
indx_word=       '/home/lvapeab/smt/tasks/emea/fren/NMT/DATA/fr.ivocab.pkl',
indx_word_target='/home/lvapeab/smt/tasks/emea/fren/NMT/DATA/en.ivocab.pkl',
word_indx=       '/home/lvapeab/smt/tasks/emea/fren/NMT/DATA/fr.vocab.pkl',
word_indx_trgt=  '/home/lvapeab/smt/tasks/emea/fren/NMT/DATA/en.vocab.pkl',

null_sym_source=70000,
null_sym_target=30000,

n_sym_source=70000 + 1,
n_sym_target=30000 + 1,

dim = 452,
rank_n_approx = 385,
encoder_stack = 1,
decoder_stack = 1,
deep_attention= False,
deep_attention_n_hids = [532, 532],

prefix='/home/lvapeab/smt/tasks/emea/fren/NMT/models/emea_385_452_',
reload = True,
overwrite = 1,
dropout = .5,
loopIters=1e9,

seqlen = 50,
sort_k_batches = 20,
bs = 80,

bleu_val_frequency=2000,
validation_burn_in=80000,
hookFreq = 1000,
validFreq = 2000,
trainFreq = 500,
saveFreq = 60,

output_validation_set=True,
validation_set='/home/lvapeab/smt/tasks/emea/DATA/test.fr',
validation_set_out='/home/lvapeab/smt/tasks/emea/fren/NMT/tmp/emea_435_532.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/emea/DATA/test.en',


#Unk Replace
unkReplace=True,
mapping = '/home/lvapeab/smt/tasks/emea/fren/NMT/DATA/topn.pkl',
heuristic=1,

# Early stop
patience = 60,
early_stop_time = 24 # In hours
)
