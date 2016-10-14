#!/usr/bin/env bash

this_dir=`pwd`;
cd ${SOFTWARE_PREFIX}/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;
cd $this_dir

# Parameters somehow "statics"
pe_script="${SOFTWARE_PREFIX}/GroundHog/experiments/nmt/online/train_online.py"

beam_size=12
src_lan="es"
trg_lan="en"
task="xerox"
v=1

model_infix="_289_354_"

state="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}state.pkl"
m1="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu50.npz"


#################################################


# Variable parameters (for experimentation)
split="test"
task="xerox"
algo="SGD"

source="${DATA_PREFIX}/${task}/DATA/${split}.${src_lan}"
refs="${DATA_PREFIX}/${task}/DATA/${split}.${trg_lan}"

for lr in 0.1; do
            echo "max_N=${max_n}"
            echo "Storing results in  ${this_dir}/${task}_${split}_${src_lan}${trg_lan}_${lr}.err "
	        echo "algo: ${algo}"
            ori_dest="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/Online/${split}.${algo}.${lr}.Orihyp.${trg_lan}"
            dest="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/Online/${split}.${algo}.${lr}.${trg_lan}"
            mkdir -p `dirname ${ori_dest}`
            mkdir -p `dirname ${dest}`
            save_ori="--save-original --save-original-to ${ori_dest}"
                python ${pe_script} --verbose ${v} --algo ${algo} --lr ${lr} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} --trans ${dest} --references ${refs} ${save_ori}"_"${lr} --models ${m1}
done
