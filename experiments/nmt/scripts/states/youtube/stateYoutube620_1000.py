dict(
#Data
source =          ['/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/300_clusters/NMT/train.vec.binarized_text.shuffled.h5'],  # Source sequences
target =          ['/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/300_clusters/NMT/train.en.binarized_text.shuffled.h5'],  # Target sequences
indx_word =        '/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/300_clusters/NMT/train.vec.ivocab.pkl',        # index -> word dict for the source language
indx_word_target = '/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/300_clusters/NMT/train.en.ivocab.pkl', # index -> word dict for the target language
word_indx =        '/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/300_clusters/NMT/train.vec.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt =   '/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/300_clusters/NMT/train.en.vocab.pkl',    # word -> index dict for the target language

source_encoding = 'utf8',
target_encoding = 'utf8',
encoding = 'utf8',

oov = 'UNK',
unk_sym_source = 1,
unk_sym_target = 1,
null_sym_source = 300,
null_sym_target = 7181,
n_sym_source =  300 + 1,
n_sym_target =  7181 + 1,

last_forward = False,
last_backward = False,
forward = True,
backward = True,                                                            
search = True,
bigram = True, # shortcut from the previous word to the current one
dec_rec_layer= 'RecurrentLayerWithSearch',
dim = 200,
rank_n_approx = 100,
maxout_part = 2.,

prefix = '/home/lvapeab/smt/tasks/video_desc/NMT/models/youtube_100_200',
reload = True,
overwrite = 1,
dropout = .5,
seqlen = 100,
sort_k_batches = 20,
bs = 80,
loopIters=1e9,

deep_attention= True,
deep_attention_n_hids = [200, 200],
deep_attention_acts= [' lambda x: T.tanh(x) ',' lambda x: T.tanh(x) '],

bleu_val_frequency=6000,
validation_burn_in=6000,
hookFreq = 2000,
validFreq = 2000,
trainFreq = 1000,
validation_set='/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/300_clusters/val.vec',
output_validation_set = True,
validation_set_out = '/home/lvapeab/smt/tasks/video_desc/NMT/tmp/dev.620-1000.hyp.en',
validation_set_grndtruth='/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/valid_corpus/val_',
saveFreq = 90,

# Early Stopping Stuff
patience = 1,
lr = 1e-3,
minlr = 0
)
