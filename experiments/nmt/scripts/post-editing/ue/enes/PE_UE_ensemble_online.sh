#!/usr/bin/env bash

this_dir=`pwd`;
cd /home/lvapeab/smt/software/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;
cd $this_dir

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble_online.py"
beam_size=12

split="test"
task="ue"
src_lan="en"
trg_lan="es"


state="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_600_420_2211_state.pkl"
m1="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_600_420_2211_model_bleu30.npz"
m2="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_600_420_2211_model_bleu33.npz"
m3="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_600_420_2211_model_bleu35.npz"
m4="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_600_420_2211_model_bleu39.npz"
m5="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ue_600_420_2211_model_bleu29.npz"

source="/home/lvapeab/smt/tasks/${task}/DATA/${split}.${src_lan}"
refs="/home/lvapeab/smt/tasks/${task}/DATA/${split}.${trg_lan}"

dest="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/postEditing/${split}.PE_isles.en"
ori_dest="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/postEditing/${split}.PE.Orihyp.en"


v=1
save_ori="--save-original --save-original-to ${ori_dest}"
mapping="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/DATA/topn.pkl"
heuristic="1"
prefix="" #"--prefix"
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
