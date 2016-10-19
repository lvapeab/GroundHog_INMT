#!/usr/bin/env bash

this_dir=`pwd`;
cd ${SOFTWARE_PREFIX}/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;
cd $this_dir

pe_script="${SOFTWARE_PREFIX}/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble.py"
beam_size=12

split="test"
task="europarl"
src_lan="fr"
trg_lan="en"

source="${DATA_PREFIX}/${task}/DATA/${split}.${src_lan}"
refs="${DATA_PREFIX}/${task}/DATA/${split}.${trg_lan}"

model_infix="_1000_620_"
state="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/NMT/models/${task}${model_infix}state.pkl"
m1="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/NMT/models/${task}${model_infix}best_bleu_model.npz"
mapping="${DATA_PREFIX}/${task}/DATA/NMT/topn_${trg_lan}.pkl"
heuristic="1"

v=1

for max_n in 6; do
	for prefix in "" "--prefix"; do
	    echo "max_N=${max_n}"
	    ori_dest="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/postEditing/${split}.PE_${prefix}.Orihyp.${trg_lan}"
        dest="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/postEditing/${split}.PE_${prefix}.${trg_lan}"
        save_ori="--save-original --save-original-to ${ori_dest}"
        log_file=${this_dir}/${task}_${split}_${src_lan}${trg_lan}_${prefix}
        echo "Storing results in ${log_file}.log and in ${log_file}.err"
        mkdir -p `dirname $dest`
        mkdir -p `dirname $ori_dest`
	    python ${pe_script} --verbose ${v} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} ${prefix} \
		--trans ${dest} --references ${refs} ${save_ori} --mapping=${mapping} --heuristic=${heuristic} --max-n ${max_n} \
		--models ${m1}  > ${log_file}.log 2>  ${log_file}.err ;
    done
done
