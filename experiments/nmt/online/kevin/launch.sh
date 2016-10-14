#!/usr/bin/env bash

if [[ $# != 9 ]]; then
    echo 'Usage: bash launch.sh source target source_file target_file state model output_prefix save_frequency learning_rate'
else
    prefix="$7_$9"
    set -x
    python /home/lvapeab/smt/software/GroundHog/experiments/nmt/online/retrain.py "$1" "$2" "$3" "$4" "$5"  -m "$6" -o  -s "$8" -l "$9" -g "SGD" &> "$prefix.out" -v
    set +x
fi
