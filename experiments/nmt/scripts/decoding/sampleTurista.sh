#!/bin/bash
#$ -l h_vmem=2g
#$ -l h_rt=8:00:00 


export THEANO_FLAGS=device=cpu,floatX=float32

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

python=/home/lvapeab/smt/software/loopy/loopy/bin/python
sampler=/home/lvapeab/smt/software/GroundHog/experiments/nmt/sample.py 
state=/home/lvapeab/nn_models/search300_250_state.pkl
beamsize=50
model=/home/lvapeab/nn_models/search300_250_model.npz
source_file=/home/lvapeab/smt/tasks/turista/NMT/DATA/test/test.es.tok
dest_file=/home/lvapeab/turista-test-300-250-Beam-50.en

${python} ${sampler} --beam-search --beam-size ${beamsize}  --state ${state} ${model}  --source ${source_file} --trans ${dest_file}