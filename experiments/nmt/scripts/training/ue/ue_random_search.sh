#!/bin/bash

trainer=/home/lvapeab/smt/software/GroundHog/experiments/nmt/train.py
task="ue"
src_lan="es"
trg_lan="en"

state=/home/lvapeab/smt/software/GroundHog/experiments/nmt/scripts/states/${task}/state.py
log_dir=/home/lvapeab/random_searches/nmt_${task}
save_dir=/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/models
tmp_dir=/home/lvapeab/smt/tasks/${task}/${src_lan}${trg_lan}/NMT/tmp


for i in `seq 1 30`;
do
    dim_word=`echo $[ 100 + $[ RANDOM % 520 ]]`
    dim=`echo $[ 100 + $[ RANDOM % 900 ]]`

    deep_attention=`echo $[ 1 + $[ RANDOM % 2 ]]`

    #case "$deep_attention" in
	#"1")
	#    deep_att="True"
	#    dropout="1.0"
	#    ;;
	#"2")
	#    deep_att="False"
	#    dropout=`echo 0.$[ 5 + $[ RANDOM % 5 ]]`
	#    ;;
	#esac
    dropout="0.5"

	###OJO QUE NO HACEMOS DEEP ATTENTION
	deep_att="False"


    prefix="${save_dir}/${task}_${dim}_${dim_word}_${deep_att}_"
    tmp="${tmp_dir}/${task}_${dim}_${dim_word}_${deep_att}.hyp.en"


    log="${log_dir}/${task}_${dim}_${dim_word}_${deep_att}.log"
    err="${log_dir}/${task}_${dim}_${dim_word}_${deep_att}.err"
    echo "`date`: Training $i started:"
    echo "Dim: $dim"
    echo "Dim_word: $dim_word"
    echo "Deep attention: $deep_att"
    echo "Dropout: $dropout"
    echo "Log dir: ${log}"
    echo "Error dir: ${err}"
    echo "Prefix: ${prefix}"

    python ${trainer} --state=${state} dim=${dim} rank_n_approx=${dim_word} dropout=${dropout} deep_attention=${deep_att} deep_attention_n_hids="[${dim}, ${dim}]" prefix="'${prefix}'" validation_set_out="'${tmp}'" > ${log} 2> ${err}

    sleep 3

    echo "`date`: Training completed"
    echo "`cat ${log} |grep Best_BLEU | tail -n 1`"
done
