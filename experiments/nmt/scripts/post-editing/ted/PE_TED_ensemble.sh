#!/usr/bin/env bash

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble.py"
beam_size=12


split="test"
task="ted"
src_lan="zh"
trg_lan="en"


state="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ted_477_547state.pkl"
m1="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ted_477_547model_bleu0.npz"
m2="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ted_477_547model_bleu1.npz"
m3="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ted_477_547model_bleu2.npz"
m4="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ted_477_547model_bleu3.npz"
m5="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models/ted_477_547model_bleu10.npz"


source="/home/lvapeab/smt/tasks/${task}/DATA/${split}.${src_lan}"
refs="/home/lvapeab/smt/tasks/${task}/DATA/${split}.${trg_lan}"



dest="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/postEditing/${split}.PE_isles.${trg_lan}"
ori_dest="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/postEditing/${split}.PE_prefix.Orihyp.${trg_lan}"


save_ori="--save-original --save-original-to ${ori_dest}"
mapping="/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/DATA/topn.pkl"
heuristic="1"

v=1
mkdir -p `dirname $dest`
mkdir -p `dirname $ori_dest`
# i="--color"
for max_n in 4; do
echo "max_N=${max_n}"
python ${pe_script} --verbose ${v} --prefix --beam-search --beam-size ${beam_size}  --state ${state} --source ${source} \
    --trans ${dest} --references ${refs} ${save_ori} --mapping=${mapping} --heuristic=${heuristic} --max-n ${max_n} ${i} \
 --models ${m1} ${m2} ${m3} ${m4} ${m5};
done
