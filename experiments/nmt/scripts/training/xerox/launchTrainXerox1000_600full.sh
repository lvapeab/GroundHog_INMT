#!/bin/bash


export OMP_NUM_THREADS=4
export THEANO_FLAGS=openmp=True



trainer=/home/lvapeab/smt/software/GroundHog_v/experiments/nmt/train.py
state=/home/lvapeab/smt/software/GroundHog_v/experiments/nmt/scripts/states/stateXerox1000_600.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
             #prototype="--proto=protoprototype_encdec_state"
skip_init=""
#skip_init="--skip-init"

${pyhton} $trainer --state=$state ${skip_init} ${prototype}