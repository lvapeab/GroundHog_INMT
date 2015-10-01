#!/bin/bash


trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train.py

state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/ue/stateUE-sel-620_500.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
#prototype="--proto=protoprototype_encdec_state"

#skipinit=""
skipinit="--skip-init"


python $trainer --state=$state ${skipinint} ${prototype}