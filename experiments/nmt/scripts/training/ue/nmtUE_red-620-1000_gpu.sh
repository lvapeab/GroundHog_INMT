#!/bin/bash

#$ -l gpu=1
#$ -l h_rt=24:00:00
#$ -l h_vmem=16g
export THEANO_FLAGS=device=gpu,floatX=float32

python=/home/lvapeab/smt/software/loopy/loopy/bin/python
trainer=/home/lvapeab/smt/software/GroundHog_v/experiments/nmt/train.py



state=/home/lvapeab/smt/software/GroundHog_v/experiments/nmt/scripts/states/ue/stateUE_sel_620_1000.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
#prototype="--proto=protoprototype_encdec_state"

#skipinit=""
skipinit="--skip-init"


${pyhton} $trainer --state=$state ${skipinint} ${prototype}