dict(
source =          ['/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/nmt_inputs/val.h5'],  # Source sequences
target =          ['/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/nmt_inputs/val.en.binarized_text.h5'],  # Target sequences
indx_word = None,        # index -> word dict for the source language
indx_word_target = '/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/nmt_inputs/val.en.ivocab.pkl', # index -> word dict for the target language
word_indx = None,         # word -> index dict for the source language
word_indx_trgt =   '/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/nmt_inputs/val.en.vocab.pkl',    # word -> index dict for the target language

source_encoding = 'utf8',
target_encoding = 'utf8',
encoding = 'utf8',

oov = 'UNK',
# unk_sym_source = 1,
unk_sym_target = 1,
# null_sym_source = 0,
null_sym_target = 372,

n_features_source =  2048,
n_sym_target =  372 + 1,

last_forward = False,
last_backward = False,
forward = True,
backward = True,
search = True,
bigram = True, # shortcut from the previous word to the current one
dec_rec_layer= 'RecurrentLayerWithSearch',
dim = 10,
rank_n_approx = 10,
maxout_part = 2.,

prefix = '/home/lvapeab/smt/tasks/video_desc/NMT/models/youtube/youtube_',
reload = False, # Ojooo
overwrite = 1,
shuffle = False,
dropout = .9,
seqlen = 500,
sort_k_batches = 2,
bs = 80,
loopIters=1e6,
use_infinite_loop = False,
videos = True,
trim_batches= False,

deep_attention= True,
deep_attention_n_hids = [10, 10],
deep_attention_acts= [' lambda x: TT.tanh(x) ',' lambda x: TT.tanh(x) '],

hookFreq = 100,
validFreq = 200,
trainFreq = 20,
saveFreq = 360,

# Early Stopping Stuff
patience = 1,
lr = 1e-3,
minlr = 0
)
