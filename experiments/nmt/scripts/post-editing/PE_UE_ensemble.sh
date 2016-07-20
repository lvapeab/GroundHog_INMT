#!/usr/bin/env bash

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble.py"
beam_size=12

state="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_600_420_2221_state.pkl"
m1="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_600_420_2221_model_bleu36.npz"
m2="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_600_420_2221_model_bleu30.npz"
m3="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_600_420_2221_model_bleu31.npz"
m4="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_600_420_2221_model_bleu38.npz"
m5="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_600_420_2221_model_bleu25.npz"

split="test"

source="/home/lvapeab/smt/tasks/ue/DATA/${split}.es"
refs="/home/lvapeab/smt/tasks/ue/DATA/${split}.en"

dest="/home/lvapeab/smt/tasks/ue/enes/NMT/postEditing/ue.${split}.PE_isles.en"
ori_dest="/home/lvapeab/smt/tasks/ue/enes/NMT/postEditing/ue.${split}.PE_isles.Orihyp.en"

v=1
save_ori="--save-original --save-original-to ${ori_dest}"
mapping="/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/topn.pkl"
heuristic="0"
# i="--color"
for max_n in 4; do
echo "max_N=${max_n}"
python ${pe_script} --verbose ${v} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} \
    --trans ${dest} --references ${refs} ${save_ori} --mapping=${mapping} --heuristic=${heuristic} --max-n ${max_n} ${i} \
 --models ${m1} ${m2} ${m3} ${m4} ${m5};
done
