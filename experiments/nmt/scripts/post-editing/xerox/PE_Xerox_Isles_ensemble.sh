#!/usr/bin/env bash

pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/postediting_sampling_isles_ensemble.py"
beam_size=12

state="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_state.pkl"
m1="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu32.npz"
m2="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu58.npz"
m3="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu17.npz"
m4="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu55.npz"
m5="/home/lvapeab/smt/tasks/xerox/esen/NMT/models/xerox_289_354_model_bleu58.npz"

split="test"
source="/home/lvapeab/smt/tasks/xerox/DATA/${split}.es"
refs="/home/lvapeab/smt/tasks/xerox/DATA/${split}.en"


#state="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_state.pkl"
#m1="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_best_bleu_model.npz"
#m2=""
#m3=""
#m4=""
#m5=""
#split="test2"
#source="/home/lvapeab/smt/tasks/xerox/DATA/lowercased/${split}.en"
#refs="/home/lvapeab/smt/tasks/xerox/DATA/lowercased/${split}.es"


dest="/home/lvapeab/smt/tasks/xerox/esen/NMT/postEditing/xerox.${split}.PE_isles.es"
ori_dest="/home/lvapeab/smt/tasks/xerox/esen/NMT/postEditing/xerox.${split}.Orihyp2.es"

v=1
save_ori="--save-original --save-original-to ${ori_dest}"
# i="--color"
for max_n in 4; do
echo "max_N=${max_n}"
python ${pe_script} --verbose ${v} --beam-search --beam-size ${beam_size} --state ${state} --source ${source} --trans ${dest} --references ${refs} ${save_ori} --max-n ${max_n} ${i} --models ${m1} ${m2} ${m3} ${m4} ${m5};
done
