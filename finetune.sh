MODEL="uncased_L-12_H-768_A-12" # BERT-Base, uncased
#MODEL="uncased_L-24_H-1024_A-16" # BERT-Large, uncased
BASEDIR=`pwd`
MODELDIR="${BASEDIR}/${MODEL}"
GLUE_BASEDIR=/om/data/public/glue/glue_data
GLUETASK="$1"
GLUEDIR="$GLUETASK"
MAX_TRAIN_STEPS=5000
OUTPUTDIR="${BASEDIR}/finetune-${MAX_TRAIN_STEPS}.${MODEL}.${GLUETASK}"

python run_classifier.py --task_name="${GLUETASK}" --do_train=true --do_eval=true \
    --data_dir="${GLUE_BASEDIR}/${GLUEDIR}" --vocab_file "${MODELDIR}/vocab.txt" \
    --bert_config_file "${MODELDIR}/bert_config.json" \
    --init_checkpoint "${MODELDIR}/bert_model.ckpt" \
    --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_steps $MAX_TRAIN_STEPS \
    --output_dir "${OUTPUTDIR}"

# symlink relevant helper data
ln -s "${MODELDIR}/bert_config.json" "${OUTPUTDIR}/bert_config.json"
ln -s "${MODELDIR}/vocab.txt" "${OUTPUTDIR}/vocab.txt"
