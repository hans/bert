INPUT=~/projects/nn-decoding/data/stimuli_384sentences.txt
#MODEL="uncased_L-12_H-768_A-12" # BERT-Base, uncased
#MODEL="uncased_L-24_H-1024_A-16" # BERT-Large, uncased
TASK="$1"
MODEL="finetune-5000.uncased_L-12_H-768_A-12.${TASK}" # BERT-Base, finetuned on MRPC
BASEDIR=`pwd`
MODELDIR="${BASEDIR}/${MODEL}"
#CKPT_NAME="bert_model" # for some reason the pretrained models have this name that's different than what the fine-tuning scripts output
CKPT_NAME="model"
CKPT_SUFFIX="$2" # can be used to load checkpoint at a particular global step
JSONL="encodings.${MODEL}.jsonl"
NPY="encodings.${MODEL}${CKPT_SUFFIX}.npy"
LAYER="-1"

# Save BERT features to jsonl file
python extract_features.py --input_file="${INPUT}" --output_file="${JSONL}" \
    --vocab_file="${MODELDIR}/vocab.txt" --bert_config_file="${MODELDIR}/bert_config.json" \
    --init_checkpoint="${MODELDIR}/${CKPT_NAME}.ckpt${CKPT_SUFFIX}" \
    --layers="-1" --max_seq_length=64 --batch_size=8 || exit 1

# Convert jsonl to common npy encoding
python process_encodings.py -i "${JSONL}" -l $LAYER -o "${NPY}"
