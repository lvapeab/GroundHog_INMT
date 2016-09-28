#!/bin/bash


trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train_unkRep.py
state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/europarl/stateFrEn.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
# prototype="--proto=prototype_search_state_with_LM_tr_en"

skipinit=""
#skipinit="--skip-init"


python $trainer --state=$state ${skipinint} ${prototype}