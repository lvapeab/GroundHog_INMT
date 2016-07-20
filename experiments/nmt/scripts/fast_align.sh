#!/usr/bin/env bash

#$ -l h_rt=12:00:00
#$ -l h_vmem=64g    

utilsdir=/home/lvapeab/smt/software/GroundHog/experiments/nmt/utils

task="ted"
src_lan="zh"
trg_lan="en"
src_vocab_size=30000
trg_vocab_size=30000
data_dir=/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/DATA


fr=/home/lvapeab/smt/tasks/${task}/DATA/training.${src_lan}
en=/home/lvapeab/smt/tasks/${task}/DATA/training.${trg_lan}

src_word2index=${data_dir}/${src_lan}.vocab.pkl
trg_word2index=${data_dir}/${trg_lan}.vocab.pkl
mapping=${data_dir}/mapping
topn=${data_dir}/topn

echo "Formatting datasets for fast_align..."
python  ${utilsdir}/format_fast_align.py ${fr} ${en} ${data_dir}/fr_en.text


echo "Aligning with fast_align..."
ttables=${data_dir}/fr_en.ttables
/home/lvapeab/smt/software/fast_align/build/fast_align -i ${data_dir}/fr_en.text -d -v -o -T 0.1 -I 4 -p ${data_dir}/fr_en.ttables > ${data_dir}/fr_en.align
echo "Corpus aligned :)"

echo "Converting alignments to top-n (1000)"
python ${utilsdir}/alignments2topn.py \
    --aligned_corpus ${data_dir}/fr_en.text \
    --alignments ${data_dir}/fr_en.align \
    --n 1000 --output ${ttables}.f1000.pkl

echo "Converting ttables..."

python ${utilsdir}/convert_Ttables.py --fname ${ttables} \
                                 --f1name ${ttables}.f1.marshal \
                                 --f1000name ${ttables}.f1000.marshal
echo "Marshalling..."
python ${utilsdir}/marshal2pkl.py ${ttables}.f1.marshal ${mapping}.pkl
python ${utilsdir}/marshal2pkl.py ${ttables}.f1000.marshal ${ttables}.f1000.pkl

echo "Creating top unigram file"

python ${utilsdir}/create_top_unigram.py --top-unigram ${ttables}.f1000.pkl --output ${topn}.pkl
# echo "Binarizing top unigram..."
# python ${utilsdir}/binarize_top_unigram.py --top-unigram ${ttables}.f1000.pkl \
#    --src-w2i ${src_word2index} --trg-w2i ${trg_word2index} \
#    --src-vocab-size ${src_vocab_size} --trg-vocab-size ${trg_vocab_size}  --output ${topn}.pkl
