MODEL="uncased_L-12_H-768_A-12" # BERT-Base, uncased
BASEDIR=`pwd`
MODELDIR="${BASEDIR}/${MODEL}"
SQUAD_DIR=/om/data/public/jgauthie/squad-2.0
MAX_TRAIN_STEPS=250
RUN="$1"
OUTPUTDIR="${BASEDIR}/finetune-${MAX_TRAIN_STEPS}.${MODEL}.SQuAD-run${RUN}"

python run_squad.py --do_train=true --do_eval=true \
    --train_file="${SQUAD_DIR}/train-v2.0.json" \
    --predict_file="${SQUAD_DIR}/dev-v2.0.json" \
    --data_dir="${SQUAD_DIR}" --vocab_file "${MODELDIR}/vocab.txt" \
    --bert_config_file "${MODELDIR}/bert_config.json" \
    --init_checkpoint "${MODELDIR}/bert_model.ckpt" \
    --save_checkpoints_steps 5 \
    --train_batch_size 12 --max_seq_length 384 --doc_stride 128 --learning_rate 3e-5 \
    --num_train_steps $MAX_TRAIN_STEPS \
    --output_dir "${OUTPUTDIR}" \
    --version_2_with_negative=True

# symlink relevant helper data
ln -s "${MODELDIR}/bert_config.json" "${OUTPUTDIR}/bert_config.json"
ln -s "${MODELDIR}/vocab.txt" "${OUTPUTDIR}/vocab.txt"
