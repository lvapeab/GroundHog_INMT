#!/usr/bin/env bash

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble.py"
beam_size=20


split="test"
task="emea"
src_lan="fr"
trg_lan="en"


state="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/emea_385_452_state.pkl"
m1="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/emea_385_452_model_bleu72.npz"
m2="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/emea_385_452_model_bleu91.npz"
m3="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/emea_385_452_model_bleu105.npz"
m4="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/emea_385_452_model_bleu125.npz"
m5="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/emea_385_452_model_bleu93.npz"


source="/home/lvapeab/smt/tasks/${task}/DATA/${split}.${src_lan}"
refs="/home/lvapeab/smt/tasks/${task}/DATA/${split}.${trg_lan}"



dest="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/postEditing/${split}.PE_isles.en"
ori_dest="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/postEditing/${split}.PE_prefix.Orihyp.en"


save_ori="--save-original --save-original-to ${ori_dest}2_isles"
mapping="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/DATA/topn.pkl"
heuristic="1"

v=1
mkdir -p `dirname $dest`
mkdir -p `dirname $ori_dest`
# i="--color"
for max_n in 4; do
echo "max_N=${max_n}"
python ${pe_script} --verbose ${v} --beam-search --beam-size ${beam_size}  --state ${state} --source ${source} \
    --trans ${dest} --references ${refs} ${save_ori} --mapping=${mapping} --heuristic=${heuristic} --max-n ${max_n} ${i} \
 --models ${m1} ${m2} ${m3} ${m4} ${m5};
done
