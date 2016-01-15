#!/bin/bash
# export THEANO_FLAGS='optimizer=None,exception_verbosity=high '
# echo "$THEANO_FLAGS"
trainer=/home/lvapeab/smt/software/GroundHog/experiments/img2txt/train.py

state=/home/lvapeab/smt/software/GroundHog/experiments/img2txt/scripts/states/youtube/state_youtube.py
prototype="" #Default: stateSearch. Si se quiere usar otro:
# prototype="--proto=prototype_search_state_with_LM_tr_en"

skipinit=""
#skipinit="--skip-init"

python $trainer --state=$state ${skipinint} ${prototype}


# python /home/lvapeab/smt/software/GroundHog/experiments/img2txt/train.py --state=/home/lvapeab/smt/software/GroundHog/experiments/img2txt/scripts/states/flickr8k/state_img.py