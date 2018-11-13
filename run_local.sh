#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 <cpu|gpu> <out_dir>" 1>&2
  exit 1
fi
BERT_TF_DIR=params/base-uncased-tf
BERT_TORCH_DIR=params/0x775cb2-base-uncased-torch
if [ "$1" == "cpu" ]
then
  export OMP_NUM_THREADS=4;
  flags="--no_cuda"
else
  flags=""
fi
python3.6 run_squad.py --vocab_file ${BERT_TF_DIR}/vocab.txt --bert_config_file ${BERT_TF_DIR}/bert_config.json --init_checkpoint ${BERT_TORCH_DIR}/pytorch_model.bin --do_predict --do_lower_case --predict_file squad/dev-v1.1.json --max_seq_length 384 --doc_stride 128 --output_dir "$2" --predict_batch_size 1 ${flags}
