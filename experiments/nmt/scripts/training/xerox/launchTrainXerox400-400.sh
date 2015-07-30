#!/bin/bash
#$ -l h_vmem=4g
#$ -l h_rt=144:00:00 

export OMP_NUM_THREADS=4
export THEANO_FLAGS=device=cpu,floatX=float32,openmp=True



python=/home/lvapeab/smt/software/loopy/loopy/bin/python
trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train.py
state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/stateXerox400_400_dec.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
             #prototype="--proto=protoprototype_encdec_state"
#skip_init=""
skip_init="--skip-init"

${pyhton} $trainer --state=$state ${skip_init} ${prototype}