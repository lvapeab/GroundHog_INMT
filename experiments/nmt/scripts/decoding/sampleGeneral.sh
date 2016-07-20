#!/bin/bash

sampler=/home/lvapeab/smt/software/GroundHog/experiments/nmt/sample_ensemble.py

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

task=xerox
split=test
src_lan="es"
dest_lan="en"
#stateN="xerox_low_250_300_state.pkl"
#model1="xerox_low_250_300_best_bleu_model.npz"
stateN="xerox_289_354_state.pkl"
model1="xerox_289_354_best_bleu_model.npz"
#model2="ue_332_289_False_model_bleu13.npz"
#model3="ue_332_289_False_model_bleu14.npz"
#model4="ue_332_289_False_model_bleu15.npz"

beamsize=5



data_dir=/home/lvapeab/smt/tasks/${task}/DATA
source_file=${data_dir}/${split}.${src_lan}
refs=${data_dir}/${split}.${dest_lan}
prefix=/home/lvapeab/smt/tasks/${task}/${src_lan}${dest_lan}/NMT/models
state=${prefix}/$stateN
m1=${prefix}/$model1
# m2=${prefix}/$model2
# m3=${prefix}/$model3
# m4=${prefix}/$model4

v=""
#dest_file=/home/lvapeab/smt/tasks/${task}/${src_lan}${dest_lan}/NMT/translations/${task}.${split}.beam_${beamsize}.hyp.${dest_lan}
dest_file=/home/lvapeab/smt/software/GroundHog/experiments/nmt/PE_isles/data/$split.en
python ${sampler} --beam-search --beam-size ${beamsize}  --state ${state}  --source ${source_file} --trans ${dest_file} ${v} --models $m1 $m2 $m3 $m4
echo "Beam: $beamsize. `thot_calc_bleu -r  ${refs} -t ${dest_file}`"

