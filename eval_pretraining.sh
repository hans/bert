#!/usr/bin/bash

# Evaluate a converged fine-tuned model on the pre-training tasks.

task=$1
run=$2
step=$3

BASEDIR=`pwd`
MODEL="uncased_L-12_H-768_A-12"
FTMODEL="${BASEDIR}/finetune-250.${MODEL}.${task}-run${run}"
LM_DATA=/om/data/public/jgauthie/books_full.dev.tfrecord

EVAL_DIR=$FTMODEL/predictions
mkdir $EVAL_DIR

echo $step
outdir=$EVAL_DIR/$step
mkdir $outdir
ln -s $FTMODEL/model.ckpt-${step}.* $outdir
echo "model_checkpoint_path: \"model.ckpt-${step}\"" > $outdir/checkpoint

python run_pretraining.py \
    --vocab_file $MODEL/vocab.txt --bert_config_file $MODEL/bert_config.json \
    --init_checkpoint=$outdir/model.ckpt-${step} \
    --eval_file=$LM_DATA \
    --do_train=True --do_eval=True \
    --eval_batch_size=32 --max_seq_length=128 --max_predictions_per_seq=20 \
    --ignore_checkpoint_variables "^output_.*" \
    --output_dir $outdir
