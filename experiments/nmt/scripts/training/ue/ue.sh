#!/bin/bash

this_dir=`pwd`;
cd ${SOFTWARE_PREFIX}/GroundHog/
python ./setup.py build install > /dev/null 2> /dev/null;
cd $this_dir

trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train_unkRep.py
state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/ue/stateEnEs.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
#prototype="--proto=prototype_search_state_with_LM_tr_en"

skipinit=""
#skipinit="--skip-init"


python $trainer --state=$state ${skipinint} ${prototype}
