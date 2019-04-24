MODEL="uncased_L-12_H-768_A-12" # BERT-Base, uncased
#MODEL="uncased_L-24_H-1024_A-16" # BERT-Large, uncased
BASEDIR=`pwd`
MODELDIR="${BASEDIR}/${MODEL}"
GLUE_BASEDIR=~/om2/data/GLUE
RUN="$1"
LM_TRAIN_DATA=/om/data/public/jgauthie/books_full.train.tfrecord
LM_DEV_DATA=/om/data/public/jgauthie/books_full.dev.tfrecord
MAX_TRAIN_STEPS=250
OUTPUTDIR="${BASEDIR}/finetune-${MAX_TRAIN_STEPS}.${MODEL}.LM-run${RUN}"

python run_pretraining.sh --do_train=true --do_eval=true \
    --vocab_file "${MODELDIR}/vocab.txt" \
    --bert_config_file "${MODELDIR}/bert_config.json" \
    --init_checkpoint "${MODELDIR}/bert_model.ckpt" \
    --train_batch_size 32 \
    --max_seq_length 128 \
    --max_predictions_per_seq 20 \
    --num_train_steps $MAX_TRAIN_STEPS \
    --num_warmup_steps 10 \
    --learning_rate 2e-5 \
    --save_checkpoints_steps 5 \
    --output_dir "${OUTPUTDIR}"

# symlink relevant helper data
ln -s "${MODELDIR}/bert_config.json" "${OUTPUTDIR}/bert_config.json"
ln -s "${MODELDIR}/vocab.txt" "${OUTPUTDIR}/vocab.txt"
