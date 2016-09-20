#!/usr/bin/env bash

this_dir=`pwd`;
cd /home/lvapeab/smt/software/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;
cd $this_dir

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble_online.py"
beam_size=12

split="test"
task="ue"
src_lan="es"
trg_lan="en"


state="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_480_480state.pkl"
m1="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_480_480model_bleu20.npz"
m2="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_480_480model_bleu24.npz"
m3="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_480_480model_bleu22.npz"
m4="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_480_480model_bleu19.npz"
m5="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_480_480model_bleu21.npz"
m6="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_480_480model_bleu25.npz"
m7="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_480_480model_bleu23.npz"
m8="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_480_480model_bleu18.npz"

source="/home/lvapeab/smt/tasks/${task}/DATA/${split}.${src_lan}"
refs="/home/lvapeab/smt/tasks/${task}/DATA/${split}.${trg_lan}"

dest="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/postEditing/${split}.PE_isles.en"
ori_dest="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/postEditing/${split}.PE.Orihyp.en"


v=1
save_ori="--save-original --save-original-to ${ori_dest}"
mapping="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/DATA/topn.pkl"
heuristic="1"
prefix="--prefix"
mkdir -p `dirname $dest`
mkdir -p `dirname $ori_dest`
# i="--color"
for max_n in 6; do
    for lr in 0.05; do	
	#for prefix in " ", "--prefix"; do
	    echo "max_N=${max_n}"
	    echo "lr=${lr}"
	    echo "Storing results in ${this_dir}/${task}_${split}_${src_lan}_${trg_lan}_${lr}.err"
	    python ${pe_script} --verbose ${v} --algo "SGD" --lr ${lr} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} ${prefix} \
		--trans ${dest} --references ${refs} ${save_ori} --mapping=${mapping} --heuristic=${heuristic} --max-n ${max_n} ${i} \
		--models ${m1} ${m2} ${m3} ${m4} ${m5} > ${this_dir}/${task}_${split}_${src_lan}_${trg_lan}_${lr}.log 2>  ${this_dir}/${task}_${split}_${src_lan}${trg_lan}_${prefix}_${lr}.err ;
	#done
    done
done
