#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 trained-model.pkl <cpu|gpu>" 1>&2
  exit 1
fi
BERT_TF_DIR=params/base-uncased-tf
BERT_TORCH_DIR=params/0x775cb2-base-uncased-torch
model_file="$1"
cpu_or_gpu="$2"
if [ "$cpu_or_gpu" == "cpu" ]
then
  export OMP_NUM_THREADS=4;
  flags="--no_cuda"
else
  flags=""
fi
python3.6 run_squad.py --vocab_file ${BERT_TF_DIR}/vocab.txt --bert_config_file ${BERT_TF_DIR}/bert_config.json --load_checkpoint "${model_file}" --run_server --do_lower_case --max_seq_length 384 --doc_stride 128 ${flags} --output_dir /tmp/bert_squad_server_out
