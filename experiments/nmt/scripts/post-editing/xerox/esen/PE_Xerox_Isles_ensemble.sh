#!/usr/bin/env bash

this_dir=`pwd`;
cd ${SOFTWARE_PREFIX}/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;
cd $this_dir

# Parameters somehow "statics"
pe_script="${SOFTWARE_PREFIX}/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble_online.py"

beam_size=12
src_lan="en"
trg_lan="es"
task="xerox"
v=1

model_infix="_289_354_"

state="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}state.pkl"
m1="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu50.npz"
m2="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu51.npz"
m3="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu8.npz"
m4="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu9.npz"
m5="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu7.npz"


#################################################


# Variable parameters (for experimentation)
split="test"
task="xerox"
algo="Adadelta"

source="${DATA_PREFIX}/${task}/DATA/${split}.${src_lan}"
refs="${DATA_PREFIX}/${task}/DATA/${split}.${trg_lan}"

for lr in 0.1; do
    for max_n in 4; do
	    for prefix in "--prefix"; do
            echo "max_N=${max_n}"
            echo "lr=${lr}"
            echo "Storing results in  ${this_dir}/${task}_${split}_${src_lan}${trg_lan}_${prefix}_${lr}.err "
            echo "max_N=${max_n}"
	    ori_dest="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/postEditing/${split}.PE_${prefix}.${algo}.${lr}.Orihyp.${trg_lan}"
	    dest="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/postEditing/${split}.PE_${prefix}.${algo}.${lr}.${trg_lan}"
	    save_ori="--save-original --save-original-to ${ori_dest}"
            python ${pe_script} ${prefix} --verbose ${v} --algo ${algo} --lr ${lr} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} --trans ${dest} --references ${refs} ${save_ori}"_"${lr} --max-n ${max_n} ${i} --models ${m1} ${m2} ${m3} ${m4} ${m5};
    	done
    done
done