#!/bin/bash

scorer="/home/lvapeab/smt/software/GroundHog/experiments/nmt/score.py"

state="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_state.pkl"
model="/home/lvapeab/smt/tasks/xerox/enes/NMT/models/xerox_fullVocab_400_520_best_bleu_model.npz"
input_sentences="/home/lvapeab/smt/tasks/emea_europarl/DATA/khresmoi-summary-dev.lowercased.en"
output_file="/home/lvapeab/reprs.txt" # If ended with .txt, the representations are stored in a raw text file. Otherwise, they are stored in hdf5 format. 


python ${scorer} --state ${state} --input ${input_sentences} --output ${output_file} --mode "save_representations" ${model}
