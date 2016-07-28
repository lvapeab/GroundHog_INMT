#!/usr/bin/env bash

this_dir=`pwd`;
cd /home/lvapeab/smt/software/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;

cd $this_dir

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble_online.py"
beam_size=12

state="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_state.pkl"
m1="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu52.npz"
m2="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu58.npz"
m3="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu51.npz"
m4="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu55.npz"
m5="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu59.npz"

split="dev"
source="/home/lvapeab/smt/tasks/xerox/DATA/${split}.es"
refs="/home/lvapeab/smt/tasks/xerox/DATA/${split}.en"


#state="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_state.pkl"
#m1="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_best_bleu_model.npz"
#m2=""
#m3=""
#m4=""
#m5=""
#split="test2"
#source="/home/lvapeab/smt/tasks/xerox/DATA/lowercased/${split}.en"
#refs="/home/lvapeab/smt/tasks/xerox/DATA/lowercased/${split}.es"


dest="/home/lvapeab/smt/tasks/xerox/esen/NMT/postEditing/xerox.${split}.PE_isles.es"
ori_dest="/home/lvapeab/smt/tasks/xerox/esen/NMT/postEditing/xerox.${split}.OrihypRetrained_SGD.es"

v=1
save_ori="--save-original --save-original-to ${ori_dest}"
# i="--color"

for lr in 1 0.5 0.1 0.05 0.01 0.005 0.0005; do
    for max_n in 4; do
        echo "max_N=${max_n}"
        python ${pe_script} --verbose ${v} --algo "SGD" --lr ${lr} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} --trans ${dest} --references ${refs} ${save_ori}"_"${lr} --max-n ${max_n} ${i} --models ${m1}  ${m2} ${m3} ${m4} ${m5};
    done
done