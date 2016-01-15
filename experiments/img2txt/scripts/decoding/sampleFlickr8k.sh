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

sampler=/home/lvapeab/smt/software/GroundHog/experiments/img2txt/sample.py
state=/home/lvapeab/smt/tasks/video_desc/NMT/models/flickr8k_3_state.pkl
beamsize=50
model=/home/lvapeab/smt/tasks/video_desc/NMT/models/flickr8k_3_model0.npz
source_file=/home/lvapeab/smt/tasks/video_desc/DATA/Flickr8k/nmt_inputs/val.h5
dest_file=/home/lvapeab/smt/tasks/video_desc/NMT/translations/val.flickr8k
nbest_file=""
aligns_file=""
v="--verbose"

python ${sampler} --beam-search --beam-size ${beamsize}  --state ${state} ${model}  --source ${source_file} --trans ${dest_file} ${v} --normalize