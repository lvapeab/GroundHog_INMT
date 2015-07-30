dict(

#Datos de entrenamiento
source = ['/home/lvapeab/smt/tasks/minieuroparl/NMT/DATA/training.en.binarized_text.h5'],  # Source sequences
target = ['/home/lvapeab/smt/tasks/minieuroparl/NMT/DATA/training.es.binarized_text.h5'],  # Target sequences    
indx_word = '/home/lvapeab/smt/tasks/minieuroparl/NMT/DATA/training.en.ivocab.pkl',        # index -> word dict for the source language
indx_word_target = '/home/lvapeab/smt/tasks/minieuroparl/NMT/DATA/training.es.ivocab.pkl', # index -> word dict for the target language
word_indx = '/home/lvapeab/smt/tasks/minieuroparl/NMT/DATA/training.en.vocab.pkl',         # word -> index dict for the source language
word_indx_trgt = '/home/lvapeab/smt/tasks/minieuroparl/NMT/DATA/training.en.vocab.pkl',    # word -> index dict for the target language

#Vocabularies
oov = 'UNK',       # String representation for the unknown word placeholder
unk_sym_source = 1,# Unknown word placeholder
unk_sym_target = 1,# Unknown word placeholder
null_sym_source = 15000,   # These are end-of-sequence marks (--> Talla del vocabulario)
null_sym_target = 15000,
n_sym_source = 15000  + 1,# These are vocabulary sizes for the source and target languages (--> talla del vocabulario +1)
n_sym_target =  15000  + 1,

#Sizes
dim = 1000, #Hidden layers dimension
rank_n_approx = 620,#Dimensionality of low-rank approximation (word-embedding)
prefix = '/home/lvapeab/nn_models/europarl620_1000_',
maxout_part = 2.,

#Model                                                                                                                                       
last_forward = False,
last_backward = False,
forward = True,
backward = True,
search = True, #"search" mechanism  
bigram = True, # shortcut from the previous word to the current one     
dec_rec_layer= 'RecurrentLayerWithSearch',# Decoder hidden layer type    

#Training details
reload = True,
overwrite = 1,
seqlen = 50,
sort_k_batches = 20,
bs = 80    )


