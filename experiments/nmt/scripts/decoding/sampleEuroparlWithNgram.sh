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


sampler=/home/lvapeab/smt/software/GroundHog/experiments/nmt/sample_withNgram.py 
state=/home/lvapeab/smt/tasks/europarl/esen/NMT/models/europarl_30k_620_1000_state.pkl
beamsize=20
model=/home/lvapeab/smt/tasks/europarl/esen/NMT/models/europarl_30k_620_1000_best_bleu_model.npz
source_file=/home/lvapeab/smt/tasks/europarl/DATA/esen/dev.es
dest_file=/home/lvapeab/smt/tasks/europarl/esen/NMT/translations/europarl.NGram.dev.en.hyp
ngram_lm=/home/lvapeab/smt/tasks/europarl/esen/LM/5gram.bin
lambda=0.8

python ${sampler} --beam-search --beam-size ${beamsize}  --state ${state} ${model}  --ngram ${ngram_lm} --lamb ${lambda}  --source ${source_file} --trans ${dest_file} ${v} 
