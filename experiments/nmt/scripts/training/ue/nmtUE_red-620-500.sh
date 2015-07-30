#!/bin/bash


trainer=/home/lvapeab/smt/software/GroundHog_v/experiments/nmt/train.py

state=/home/lvapeab/smt/software/GroundHog_v/experiments/nmt/scripts/states/ue/stateUE_sel_620_500.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
#prototype="--proto=protoprototype_encdec_state"

skipinit=""
#skipinit="--skip-init"


python $trainer --state=$state ${skipinint} ${prototype}