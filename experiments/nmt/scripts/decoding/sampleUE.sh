#!/bin/bash
#$ -l h_vmem=16g
#$ -l h_rt=18:00:00 


export THEANO_FLAGS=on_unused_input='warn'

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


sampler=/home/lvapeab/smt/software/GroundHog/experiments/nmt/sample_pp.py
state=/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_11k_full_620_500_dropstate.pkl
beamsize=80
model=/home/lvapeab/smt/tasks/ue/esen/NMT/models/ue_11k_full_620_500_dropbest_bleu_model.npz
source_file=/home/lvapeab/smt/tasks/ue/esen/DATA/test.es
dest_file=/home/lvapeab/smt/tasks/ue/esen/NMT/translations/uefull-test-620-500.en
nbest_file=""
aligns_file=""
v=""

python ${sampler} --beam-search --beam-size ${beamsize}  --state ${state} ${model}  --source ${source_file} --trans ${dest_file} ${v} \
# --normalize --print-aligns ${aligns_file} # --nbest 50 --nbest-dest ${nbest_file}ource_file} --trans ${dest_file} ${v} # --normalize