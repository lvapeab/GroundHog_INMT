#!/bin/bash


trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train.py
state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/turista/state.py
prototype="" #Default: stateSearch. Si se quiere usar otro:                                                                                  
             #prototype="--proto=protoprototype_encdec_state"                                                                                              
skip_init="" #--skip-init

python $trainer --state=$state ${skip_init}  ${prototype}
