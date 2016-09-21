#!/usr/bin/env bash

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble.py"
beam_size=12

state="/media/HDD_2TB/MODELS/xerox/esen/models/xerox_289_354_state.pkl"
m1="/media/HDD_2TB/MODELS/xerox/esen/models/xerox_289_354_model_bleu58.npz"
m2="/media/HDD_2TB/MODELS/xerox/esen/models/xerox_289_354_model_bleu59.npz"
m3="/media/HDD_2TB/MODELS/xerox/esen/models/xerox_289_354_model_bleu55.npz"

split="test"
source="/media/HDD_2TB/DATASETS/xerox/DATA/${split}.es"
refs="/media/HDD_2TB/DATASETS/xerox/DATA/${split}.en"


dest="/media/HDD_2TB/MODELS/xerox/esen//postEditing/new_xerox.${split}.PE_isles.es"
ori_dest="/media/HDD_2TB/MODELS/xerox/esen//postEditing/new_xerox.${split}.Orihyp2.es"

v=1
save_ori="--save-original --save-original-to ${ori_dest}"
# i="--color"
for max_n in 4; do
echo "max_N=${max_n}"
python ${pe_script} --verbose ${v} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} --trans ${dest} --references ${refs} ${save_ori} --max-n ${max_n} ${i} --models ${m1} ${m2} ${m3} # ${m4} ${m5};
done
