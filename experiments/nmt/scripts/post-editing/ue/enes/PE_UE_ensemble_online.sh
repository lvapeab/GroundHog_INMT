#!/usr/bin/env bash

this_dir=`pwd`;
cd ${SOFTWARE_PREFIX}/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;
cd $this_dir

pe_script="${SOFTWARE_PREFIX}/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble.py"
beam_size=12

split="test"
task="ue"
src_lan="en"
trg_lan="es"
source="${DATA_PREFIX}/${task}/DATA/${split}.${src_lan}"
refs="${DATA_PREFIX}/${task}/DATA/${split}.${trg_lan}"

state="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/ue_600_420_2211_state.pkl"
m1="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/ue_600_420_2211_model_bleu30.npz"
m2="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/ue_600_420_2211_model_bleu33.npz"
m3="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/ue_600_420_2211_model_bleu35.npz"
m4="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/ue_600_420_2211_model_bleu39.npz"
m5="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/ue_600_420_2211_model_bleu29.npz"
m6="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/ue_600_420_2211_model_bleu31.npz"
m7="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/ue_600_420_2211_model_bleu32.npz"
m8="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/ue_600_420_2211_model_bleu34.npz"
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
		--models ${m1} ${m2} ${m3} ${m4} ${m5} ${m6} ${m7} ${m8} > ${log_file}.log 2>  ${log_file}.err ;
    done
done
