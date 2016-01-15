#!/bin/bash


#$ -l gpu=1 -l  h_vmem=32g,h_rt=24:00:00

trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train.py
state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/europarl/stateEmeaEuroparl520_800_sameLang.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
# prototype="--proto=prototype_search_state_with_LM_tr_en"

#skipinit=""
skipinit="--skip-init"


python $trainer --state=$state ${skipinint} ${prototype}