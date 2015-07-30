#!/bin/bash
#$ -l h_vmem=8g
#$ -l h_rt=144:00:00 


export THEANO_FLAGS=device=cpu,floatX=float32,openmp=True

python=/home/lvapeab/smt/software/loopy/loopy/bin/python
trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train.py
state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/stateEuroparl620_1000.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
#prototype="--proto=protoprototype_encdec_state"


${pyhton} $trainer --state=$state --skip-init ${prototype}