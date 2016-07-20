#!/bin/bash
trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train_unkRep.py
state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/emea/stateEmeaFrEn.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
             #prototype="--proto=protoprototype_encdec_state"
skip_init=""
#skip_init="--skip-init"

python $trainer --state=$state ${skip_init} ${prototype}