#!/usr/bin/env bash

this_dir=`pwd`;
cd ${SOFTWARE_PREFIX}/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;
cd $this_dir

# Parameters somehow "statics"
pe_script="${SOFTWARE_PREFIX}/GroundHog/experiments/nmt/online/train_online.py"


beamsize=12
src_lan="fr"
trg_lan="en"
task="europarl"
v=1

split="dev"

model_infix="_1000_620_"

state="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}state.pkl"
m1="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}best_bleu_model.npz"
m2="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu6.npz"
m3="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu50.npz"
m4="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu59.npz"
m5="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/models/${task}${model_infix}model_bleu62.npz"
#################################################


# Variable parameters (for experimentation)
source=${DATA_PREFIX}/${task}/DATA/${split}.${src_lan}
refs=${DATA_PREFIX}/${task}/DATA/${split}.${trg_lan}
topn="${DATA_PREFIX}/${task}/DATA/NMT/topn_${trg_lan}.pkl"

n_iter=1
for algo in "AdaGrad" ; do
    for lr in 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001 ; do
            echo "algo: ${algo}"
            dest="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/Online/${split}.${task}.${task}.${algo}.${lr}.${trg_lan}"
            mkdir -p `dirname ${dest}`
            python ${pe_script} --verbose ${v} --replaceUnk --mapping ${topn} --heuristic 1 --algo ${algo} --lr ${lr} --beam-search --beam-size ${beamsize} --state ${state} --source ${source} --trans ${dest} --refs ${refs} --n-iters ${n_iter} --models ${m1} # ${m2} ${m3} ${m4} ${m5}
        for n_iter in `seq 1 $n_iter`; do
            echo "BLEU: `calc_bleu -r  ${refs} -t ${dest}.iter_${n_iter}`"
        done
    done
done

for algo in "SGD" ; do
    for lr in 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001 ; do
            echo "algo: ${algo}"
            dest="${MODELS_PREFIX}/${task}/${src_lan}${trg_lan}/Online/${split}.${task}.${task}.${algo}.${lr}.${trg_lan}"
            mkdir -p `dirname ${dest}`
            python ${pe_script} --verbose ${v} --replaceUnk --mapping ${topn} --heuristic 1 --algo ${algo} --lr ${lr} --beam-search --beam-size ${beamsize} --state ${state} --source ${source} --trans ${dest} --refs ${refs} --n-iters ${n_iter} --models ${m1} # ${m2} ${m3} ${m4} ${m5}
        for n_iter in `seq 1 $n_iter`; do
            echo "BLEU: `calc_bleu -r  ${refs} -t ${dest}.iter_${n_iter}`"
        done
    done
done
