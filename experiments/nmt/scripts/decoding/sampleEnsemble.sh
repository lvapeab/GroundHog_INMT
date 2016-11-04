#!/bin/bash
sampler=${SOFTWARE_PREFIX}/GroundHog/experiments/nmt/sample_ensemble.py


beamsize=12
src_lan="en"
trg_lan="es"
task="xerox"
split="dev"

v=1

source_file=${DATA_PREFIX}/$task/DATA/${split}.${src_lan}
refs=${DATA_PREFIX}/$task/DATA/${split}.${trg_lan}

model_infix="_600_420_2211_"

state="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}state.pkl"
m1="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}best_bleu_model.npz"
m2="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu6.npz"
m3="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu50.npz"
m4="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu59.npz"
m5="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu62.npz"


echo "Beam_size: $beamsize"
dest="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/Online/${split}.OFFLINE.ensemble.${trg_lan}"
mkdir -p `dirname $dest`

time python ${sampler} --beam-search --beam-size ${beamsize} --notReplaceUnk --state ${state}  --source ${source_file} --trans ${dest} --models ${m1} # ${m2} ${m3} ${m4} ${m5}

echo "Beam: $beamsize. `calc_bleu -r  ${refs} -t ${dest}`"
