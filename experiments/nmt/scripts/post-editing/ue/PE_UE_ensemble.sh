#!/usr/bin/env bash

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble.py"
beam_size=20

state="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_480_480state.pkl"
m1="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_480_480model_bleu20.npz"
m2="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_480_480model_bleu24.npz"
m3="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_480_480model_bleu22.npz"
m4="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_480_480model_bleu19.npz"
m5="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_480_480model_bleu21.npz"
m6="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_480_480model_bleu25.npz"
m7="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_480_480model_bleu23.npz"
m8="/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_480_480model_bleu18.npz"

split="test"

source="/home/lvapeab/smt/tasks/ue/DATA/${split}.es"
refs="/home/lvapeab/smt/tasks/ue/DATA/${split}.en"

dest="/home/lvapeab/smt/tasks/ue/esen/NMT/postEditing/ue.${split}.PE_isles_prefix.en"
ori_dest="/home/lvapeab/smt/tasks/ue/esen/NMT/postEditing/ue.${split}.PE_isles.Orihyp2_prefix.en"

v=1
save_ori="--save-original --save-original-to ${ori_dest}"
mapping="/home/lvapeab/smt/tasks/ue/esen/NMT/DATA/topn.pkl"
heuristic="1"
# i="--color"
for max_n in 7; do
echo "max_N=${max_n}"
python ${pe_script} --verbose ${v} --prefix --beam-search --beam-size ${beam_size} --state ${state} --source ${source} \
    --trans ${dest} --references ${refs} ${save_ori} --mapping=${mapping} --heuristic=${heuristic} --max-n ${max_n} ${i} \
 --models ${m1} ${m2} ${m3} ${m4} ${m5};
done
