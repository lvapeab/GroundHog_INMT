#!/usr/bin/env bash


pe_script="/home/lvapeab/smt/software/GroundHog/experiments/nmt/postediting_sampling.py"
beam_size=6

state="/home/lvapeab/smt/tasks/europarl/esen/NMT/models/europarl_30k_620_1000_state.pkl"
model="/home/lvapeab/smt/tasks/europarl/esen/NMT/models/europarl_30k_620_1000_best_bleu_model.npz"

source="/home/lvapeab/smt/tasks/europarl/DATA/esen/dev.es"
dest="/home/lvapeab/smt/tasks/europarl/esen/NMT/translations/europarl.dev.PE.en"
refs="/home/lvapeab/smt/tasks/europarl/DATA/esen/dev.en"
v=""
# v="--verbose"

python ${pe_script} --beam-search --beam-size ${beam_size} --state ${state} ${model} --source ${source} --trans ${dest} --references ${refs} ${v}
