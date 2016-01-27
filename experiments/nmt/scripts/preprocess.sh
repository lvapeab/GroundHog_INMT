#!/bin/bash

scriptsdir=/home/lvapeab/smt/software/GroundHog/experiments/nmt/preprocess
mosesdir=/home/lvapeab/smt/software/mosesdecoder


#Input corpora

# Xerox
#sourceText=/home/lvapeab/smt/tasks/xerox/enes/DATA/training.en
#targetText=/home/lvapeab/smt/tasks/xerox/enes/DATA/training.es


# UE
sourceText=/home/lvapeab/smt/tasks/ue/DATA/training.es
targetText=/home/lvapeab/smt/tasks/ue/DATA/training.en

# UE
sourceNameDest=/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/es
targetNameDest=/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/en


#Short-list length (vocabulary size)

v_size_src=30000
v_size_trg=30000

#Validation set size ( If >= 1 take this many samples for the validation set, if < 1, take this fraction of the samples)

#split="--split 0.1 --ngram 4"
split=""

tokenize=0 # Tokenize data?
#tok="tok"     # Tokenized files suffix
tok=""
mkdir -p `dirname $sourceNameDest`
mkdir -p `dirname $targetNameDest`



if [ ${tokenize} -eq 1 ]; then
    #Tokenization
    echo "Tokenizing corpus..."
    tok=".tok"
    perl ${mosesdir}/scripts/tokenizer/tokenizer.perl -l en < ${sourceText} > ${sourceNameDest}${tok}
    perl ${mosesdir}/scripts/tokenizer/tokenizer.perl -l es < ${targetText} > ${targetNameDest}${tok}
else
    echo "We won't tokenize the corpus"
    cat ${sourceText} > ${sourceNameDest}${tok}
    cat ${targetText} > ${targetNameDest}${tok}
    
fi



echo "==== Preprocessing source corpus ==="
python ${scriptsdir}/preprocess.py --overwrite --dictionary ${sourceNameDest}.vocab.pkl --vocab $v_size_src --binarized-text ${sourceNameDest}.binarized_text.pkl --pickle ${sourceNameDest}${tok} ${split}

echo "===================================="
echo "==== Preprocessing target corpus ==="

python ${scriptsdir}/preprocess.py --overwrite --dictionary ${targetNameDest}.vocab.pkl --vocab $v_size_trg --binarized-text ${targetNameDest}.binarized_text.pkl --pickle ${targetNameDest}${tok}  ${split}


echo "===================================="
echo "==== Inverting vocabulary dictionaries ==="

python ${scriptsdir}/invert-dict.py ${sourceNameDest}.vocab.pkl ${sourceNameDest}.ivocab.pkl 
python ${scriptsdir}/invert-dict.py ${targetNameDest}.vocab.pkl ${targetNameDest}.ivocab.pkl 


echo "===================================="
echo "==== Converting to hdf5 format ==="

python ${scriptsdir}/convert-pkl2hdf5.py ${sourceNameDest}.binarized_text.pkl ${sourceNameDest}.binarized_text.h5 
python ${scriptsdir}/convert-pkl2hdf5.py ${targetNameDest}.binarized_text.pkl ${targetNameDest}.binarized_text.h5 


echo "===================================="
echo "==== Shuffling ==="

python ${scriptsdir}/shuffle-hdf5.py ${sourceNameDest}.binarized_text.h5 ${targetNameDest}.binarized_text.h5 ${sourceNameDest}.binarized_text.shuffled.h5 ${targetNameDest}.binarized_text.shuffled.h5


echo "===================================="
echo "==== Preprocessing finished ==="
echo "===================================="
