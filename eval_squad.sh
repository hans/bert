#!/usr/bin/bash

run=$1
step=$2

BASEDIR=`pwd`
MODEL="uncased_L-12_H-768_A-12"
FTMODEL="${BASEDIR}/finetune-250.${MODEL}.SQuAD-run${run}"
SQUADDIR=/om/data/public/jgauthie/squad-2.0

PREDDIR=$FTMODEL/predictions
mkdir $PREDDIR

echo $step
outdir=$PREDDIR/$step
mkdir $outdir
ln -s $FTMODEL/model.ckpt-${step}.* $outdir
echo "model_checkpoint_path: \"model.ckpt-${step}\"" > $outdir/checkpoint

python run_squad.py --do_predict \
    --vocab_file $MODEL/vocab.txt --bert_config_file $MODEL/bert_config.json \
    --init_checkpoint=$outdir/model.ckpt-${step} \
    --predict_file=$SQUADDIR/dev-v2.0.json \
    --doc_stride 128 --version_2_with_negative=True \
    --predict_batch_size 32 \
    --output_dir $outdir

python $SQUADDIR/evaluate-v2.0.py $SQUADDIR/dev-v2.0.json $outdir/predictions.json --na-prob-file $outdir/null_odds.json > $outdir/results.json

rm $outdir/eval.tf_record
