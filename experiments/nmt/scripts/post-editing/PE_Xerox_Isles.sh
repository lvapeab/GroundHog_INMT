#!/usr/bin/env bash


pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles.py"
beam_size=6

state="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_state.pkl"
model="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_best_bleu_model.npz"


split="test"

source="/home/lvapeab/smt/tasks/xerox/enes/DATA/${split}.en"
dest="/home/lvapeab/smt/tasks/xerox/enes/NMT/postEditing/xerox.${split}.PE_isles.es"
ori_dest="/home/lvapeab/smt/tasks/xerox/enes/NMT/postEditing/xerox.${split}.Orihyp.es"
refs="/home/lvapeab/smt/tasks/xerox/enes/DATA/${split}.es"

#v="--verbose"
save_ori="--save-original --save-original-to ${ori_dest}"
# i="--color"
max_n=5


for max_n in 4; do
echo "max_N=${max_n}"
python ${pe_script} ${v} --beam-search --beam-size ${beam_size} --state ${state}  --source ${source} --trans ${dest} --references ${refs} ${save_ori} --max-n ${max_n} ${i} ${model};
done
