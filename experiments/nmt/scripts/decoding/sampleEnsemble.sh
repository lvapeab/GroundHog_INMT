#!/bin/bash
#$ -l gpu=1,h_vmem=64g,h_rt=4:00:00

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
data_dir=/home/lvapeab/smt/tasks/${task}/DATA
split=dev
src_lan="es"
dest_lan="en"

source_file=${data_dir}/${split}.${src_lan}
refs=${data_dir}/${split}.${dest_lan}

model_name="xerox_289_354_"
prefix=/home/lvapeab/smt/tasks/${task}/${src_lan}${dest_lan}/NMT/models/${model_name}
beamsize=6

state=${prefix}state.pkl
m1=${prefix}model_bleu32.npz
#m2=${prefix}model_bleu58.npz
#m3=${prefix}model_bleu17.npz
#m4=${prefix}model_bleu55.npz
#m5=${prefix}model_bleu58.npz


v=""

dest_file=/home/lvapeab/smt/tasks/xerox/enes/NMT/translations/xerox.${split}.beam_${beam_size}.hyp.${dest_lan}
python ${sampler} --beam-search --beam-size ${beamsize}  --state ${state}  --source ${source_file} --trans ${dest_file} ${v} --models ${m1} #${m2} ${m3} ${m4} ${m5}

echo "Beam: $beamsize. `thot_calc_bleu -r  ${refs} -t ${dest_file}`"

