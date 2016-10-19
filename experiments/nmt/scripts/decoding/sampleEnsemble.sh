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

task=europarl
split=test
src_lan="en"
dest_lan="fr"

source_file=${DATA_PREFIX}/$task/DATA/${split}.${src_lan}
refs=${DATA_PREFIX}/$task/DATA//${split}.${dest_lan}

model_name="europarl_1000_620_"
prefix=${MODELS_PREFIX}/${task}/${src_lan}${dest_lan}/NMT/models/${model_name}
beamsize=12

state=${prefix}state.pkl
m1=${prefix}model_bleu98.npz
m2=${prefix}model_bleu103.npz
m3=${prefix}model_bleu105.npz

v=""
for beamsize in 12 20; 
do
    echo "Beam_size: $beamsize"
    dest_file=${MODELS_PREFIX}/${task}/${src_lan}${dest_lan}/NMT/translations/${task}.${split}.beam_${beamsize}.hyp.${dest_lan}
    time python ${sampler} --beam-search --beam-size ${beamsize}  --state ${state}  --source ${source_file} --trans ${dest_file} ${v} --models ${m1} ${m2} ${m3} # ${m4} ${m5}

    echo "Beam: $beamsize. `thot_calc_bleu -r  ${refs} -t ${dest_file}`"
done
