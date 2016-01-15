#!/bin/bash

scriptsdir=/home/lvapeab/smt/software/GroundHog/experiments/nmt/preprocess
mosesdir=/home/lvapeab/smt/software/mosesdecoder


#Input corpora

# MSRV

sourceText=""
targetText=/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/nmt_inputs/val.en

#Output files

# MSRV

sourceNameDest=""
targetNameDest=/home/lvapeab/smt/tasks/video_desc/DATA/MSRVD/nmt_inputs/val.en


# Number of features of each vector
f_vector_size=2048

#Short-list length (vocabulary size)
v_size_trg=372

#Validation set size ( If >= 1 take this many samples for the validation set, if < 1, take this fraction of the samples)

#split="--split 0.1 --ngram 4"
split=""

#mkdir -p `dirname $sourceNameDest`
#mkdir -p `dirname $targetNameDest`





echo "==== Preprocessing source corpus ==="
# python ${scriptsdir}/preprocess_feature_vectors.py --overwrite -d $f_vector_size --binarized-text ${sourceNameDest}.binarized_vectors.pkl ${sourceNameDest}

echo "===================================="
echo "==== Preprocessing target corpus ==="

python ${scriptsdir}/preprocess.py --overwrite --dictionary ${targetNameDest}.vocab.pkl --vocab $v_size_trg --binarized-text ${targetNameDest}.binarized_text.pkl --pickle ${targetText}  ${split}


echo "===================================="
echo "==== Inverting vocabulary dictionaries ==="

python ${scriptsdir}/invert-dict.py ${targetNameDest}.vocab.pkl ${targetNameDest}.ivocab.pkl 


echo "===================================="
echo "==== Converting to hdf5 format ==="

# python ${scriptsdir}/convert-pkl2hdf5.py ${sourceNameDest}.binarized_vectors.pkl ${sourceNameDest}.binarized_vectors.h5
python ${scriptsdir}/convert-pkl2hdf5.py ${targetNameDest}.binarized_text.pkl ${targetNameDest}.binarized_text.h5 


echo "===================================="
echo "==== Shuffling? ==="

echo "We don't shuffle... for the moment"
# python ${scriptsdir}/shuffle-hdf5.py ${sourceNameDest}.binarized_text.h5 ${targetNameDest}.binarized_text.h5 ${sourceNameDest}.binarized_text.shuffled.h5 ${targetNameDest}.binarized_text.shuffled.h5


echo "===================================="
echo "==== Preprocessing finished ==="
echo "===================================="