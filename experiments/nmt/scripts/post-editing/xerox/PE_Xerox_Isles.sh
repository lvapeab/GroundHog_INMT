#!/usr/bin/env bash

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles.py"
beam_size=6

state="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_state.pkl"
model="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_best_bleu_model.npz"


split="test"

source="/home/lvapeab/smt/tasks/xerox/DATA/lowercased/${split}.en"
refs="/home/lvapeab/smt/tasks/xerox/DATA/lowercased/${split}.es"

dest="/home/lvapeab/smt/tasks/xerox/enes/NMT/postEditing/xerox.${split}.PE_isles.es"
ori_dest="/home/lvapeab/smt/tasks/xerox/enes/NMT/postEditing/xerox.${split}.Orihyp_singleNN.es"

v="1"
save_ori="--save-original --save-original-to ${ori_dest}"
# i="--color"
for max_n in 4; do
echo "max_N=${max_n}"
python ${pe_script} --verbose ${v} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} --trans ${dest} --references ${refs} ${save_ori} --max-n ${max_n} ${i} ${model};
done
