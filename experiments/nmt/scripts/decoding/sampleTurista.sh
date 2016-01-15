#!/bin/bash



export THEANO_FLAGS=device=cpu,floatX=float32,mode=FAST_COMPILE

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
state=/home/lvapeab/smt/tasks/turista/NMT/models/turista_250_300_state.pkl
beamsize=20
model=/home/lvapeab/smt/tasks/turista/NMT/models/turista_250_300_model.npz
source_file=/home/lvapeab/smt/tasks/turista/DATA/test.es
dest_file=/home/lvapeab/smt/tasks/turista/NMT/tmp/test.NonInter.en.hyp


python ${sampler} --beam-search --beam-size ${beamsize} --state ${state} ${model}  --source ${source_file} --trans ${dest_file}
