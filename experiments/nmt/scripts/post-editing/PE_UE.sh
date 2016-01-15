#!/usr/bin/env bash


pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/postediting_sampling.py"
beam_size=10

state="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_11k_full_620_500_state.pkl"
model="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_11k_full_620_500_best_bleu_model.npz"

source="/home/lvapeab/smt/tasks/ue/DATA/test.es"
dest="/home/lvapeab/smt/tasks/ue/esen/NMT/postEditing/ue.test.PE.en"
refs="/home/lvapeab/smt/tasks/ue/DATA/test.en"



#v="--verbose"
v=""
ori_dest="/home/lvapeab/smt/tasks/ue/esen/NMT/postEditing/ue.test.hyp.en"
save_ori="--save-original --save-original-to ${ori_dest}"


python ${pe_script} ${v} --beam-search --beam-size ${beam_size} --state ${state}  --source ${source} --trans ${dest} --references ${refs} ${save_ori} ${model}
