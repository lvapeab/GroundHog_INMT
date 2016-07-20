#!/usr/bin/env bash


pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/postediting_sampling.py"
beam_size=6

state="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_state.pkl"
model="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_best_bleu_model.npz"

source="/home/lvapeab/smt/tasks/xerox/enes/DATA/dev.en"
dest="/home/lvapeab/smt/tasks/xerox/enes/NMT/postEditing/xerox.dev.PE_prefixes.en"
ori_dest="/home/lvapeab/smt/tasks/xerox/enes/NMT/postEditing/xerox.dev.hyp.en"
refs="/home/lvapeab/smt/tasks/xerox/enes/DATA/dev.es"

# v="--verbose"
v=""
save_ori="--save-original --save-original-to ${ori_dest}"

python ${pe_script} ${v} --beam-search --beam-size ${beam_size} --state ${state}  --source ${source} --trans ${dest} --references ${refs} ${save_ori} ${model}
