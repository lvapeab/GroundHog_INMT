#!/bin/bash


#usage: Sample (of find with beam-serch) translations from a translation model
#       [-h] --state STATE [--beam-search] [--beam-size BEAM_SIZE]
#       [--ignore-unk] [--source SOURCE] [--trans TRANS] [--normalize]
#       [--verbose]
#       model_path [changes]
#
#positional arguments:
#  model_path            Path to the model
#  changes               Changes to state
#
#optional arguments:
#  -h, --help            show this help message and exit
#  --state STATE         State to use
#  --beam-search         Beam size, turns on beam-search
#  --beam-size BEAM_SIZE
#                        Beam size
#  --ignore-unk          Ignore unknown words
#  --source SOURCE       File of source sentences
#  --trans TRANS         File to save translations in
#  --normalize           Normalize log-prob with the word count
#  --verbose             Be verbose


sampler=/home/lvapeab/smt/software/GroundHog/experiments/nmt/sample.py
state=/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_state.pkl
model=/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_best_bleu_model.npz

split=test
source_file=/home/lvapeab/smt/tasks/xerox/enes/DATA/${split}.en
refs=/home/lvapeab/smt/tasks/xerox/enes/DATA/${split}.es
bleu=

v=""

for beam_size in 1 2 4 6 8 10 12 20; do
dest_file=/home/lvapeab/smt/tasks/xerox/enes/NMT/translations/xerox.${split}.beam_${beam_size}.hyp.es
python ${sampler} --beam-search --beam-size ${beam_size}  --state ${state} ${model}  --source ${source_file} --trans ${dest_file} ${v}

echo "Beam: $beam_size. `thot_calc_bleu -r  ${refs} -t ${dest_file}`"

done