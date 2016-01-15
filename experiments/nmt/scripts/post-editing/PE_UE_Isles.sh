#!/usr/bin/env bash


pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles.py"
beam_size=10
state="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_11k_full_620_500_state.pkl"
model="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_11k_full_620_500_best_bleu_model.npz"

split="test"

source="/home/lvapeab/smt/tasks/ue/DATA/${split}.es"
dest="/home/lvapeab/smt/tasks/ue/esen/NMT/postEditing/ue.${split}.PE.en"
ori_dest="/home/lvapeab/smt/tasks/ue/esen/NMT/postEditing/ue.${split}.Orihyp.es"
refs="/home/lvapeab/smt/tasks/ue/DATA/${split}.en"

#v="--verbose"
save_ori="--save-original --save-original-to ${ori_dest}"
# i="--color"
max_n=7


for max_n in `seq 1 15`; do
echo "max_N=${max_n}"
python ${pe_script} ${v} --beam-search --beam-size ${beam_size} --state ${state}  --source ${source} --trans ${dest} --references ${refs} ${save_ori} --max-n ${max_n} ${i} ${model};
done
