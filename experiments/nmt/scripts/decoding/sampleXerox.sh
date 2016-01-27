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
data_dir=/home/lvapeab/smt/tasks/xerox/DATA/original
split="dev"
src_lan="en"
dest_lan="es"

prefix="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_TrueCased_420_500_"

state=${prefix}state.pkl
model=${prefix}best_bleu_model.npz

source_file=${data_dir}/${split}.${src_lan}
refs=${data_dir}/${split}.${dest_lan}
v=""

beamsize=6
dest_file=/home/lvapeab/smt/tasks/xerox/enes/NMT/translations/xerox.${split}.beam_${beam_size}.hyp.${dest_lan}
python ${sampler} --beam-search --beam-size ${beamsize}  --state ${state} ${model}  --source ${source_file} --trans ${dest_file} ${v}

echo "Beam: $beam_size. `thot_calc_bleu -r  ${refs} -t ${dest_file}`"

