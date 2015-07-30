#!/bin/bash
#$ -l h_vmem=8g
#$ -l h_rt=144:00:00 
# $ -pe mp 4
# $ -v OMP_NUM_THREADS=4

#export OMP_NUM_THREADS=4
export THEANO_FLAGS=device=cpu,floatX=float32,openmp=True



python=/home/lvapeab/smt/software/loopy/loopy/bin/python
trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train.py

state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/xerox/stateXerox620_400_15kIter.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
#prototype="--proto=protoprototype_encdec_state"

skipinit=""
#skipinit="--skip-init"


${pyhton} $trainer --state=$state ${skipinint} ${prototype}