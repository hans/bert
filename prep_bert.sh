INPUT=~/projects/nn-decoding/data/stimuli_384sentences.txt
#MODEL="uncased_L-12_H-768_A-12" # BERT-Base, uncased
MODEL="uncased_L-24_H-1024_A-16" # BERT-Large, uncased
BASEDIR=`pwd`
MODELDIR="${BASEDIR}/${MODEL}"
JSONL="encodings.${MODEL}.jsonl"
NPY="encodings.${MODEL}.npy"
LAYER="-1"

# Save BERT features to jsonl file
python extract_features.py --input_file="${INPUT}" --output_file="${JSONL}" \
    --vocab_file="${MODELDIR}/vocab.txt" --bert_config_file="${MODELDIR}/bert_config.json" \
    --init_checkpoint="${MODELDIR}/bert_model.ckpt" \
    --layers="-1" --max_seq_length=64 --batch_size=8

# Convert jsonl to common npy encoding
python process.py -i "${JSONL}" -l $LAYER -o "${NPY}"
