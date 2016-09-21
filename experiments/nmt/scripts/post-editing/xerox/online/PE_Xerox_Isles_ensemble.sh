#!/usr/bin/env bash

this_dir=`pwd`;
cd /home/lvapeab/smt/software/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;
cd $this_dir

# Parameters somehow "statics"
pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble_online.py"
models_prefix="/media/HDD_2TB/MODELS"
data_prefix="/media/HDD_2TB/DATASETS"

beam_size=12
src_lan="en"
trg_lan="es"
task="xerox"
v=1

model_infix="_600_420_2211_"

state="${models_prefix}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}state.pkl"
m1="${models_prefix}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu98.npz"
m2="${models_prefix}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu90.npz"
m3="${models_prefix}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu95.npz"
m4="${models_prefix}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu96.npz"
m5="${models_prefix}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu94.npz"


#################################################


# Variable parameters (for experimentation)
split="test"
task="xerox"
algo="Adadelta"

source="${data_prefix}/${task}/DATA/${split}.${src_lan}"
refs="${data_prefix}/${task}/DATA/${split}.${trg_lan}"

for lr in 0.1; do
    for max_n in 4; do
	    for prefix in "--prefix"; do
            echo "max_N=${max_n}"
            echo "lr=${lr}"
            echo "Storing results in  ${this_dir}/${task}_${split}_${src_lan}${trg_lan}_${prefix}_${lr}.err "
            echo "max_N=${max_n}"
	    ori_dest="${models_prefix}/${task}/${src_lan}${trg_lan}/postEditing/${split}.PE_${prefix}.${algo}.${lr}.Orihyp.${trg_lan}"
	    dest="${models_prefix}/${task}/${src_lan}${trg_lan}/postEditing/${split}.PE_${prefix}.${algo}.${lr}.${trg_lan}"
	    save_ori="--save-original --save-original-to ${ori_dest}"
            python ${pe_script} ${prefix} --verbose ${v} --algo ${algo} --lr ${lr} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} --trans ${dest} --references ${refs} ${save_ori}"_"${lr} --max-n ${max_n} ${i} --models ${m1} ;
    	done
    done
done
