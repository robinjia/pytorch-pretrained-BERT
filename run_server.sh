#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 trained-model.pkl <cpu|gpu>" 1>&2
  exit 1
fi
model_file="$1"
cpu_or_gpu="$2"
if [ "$cpu_or_gpu" == "cpu" ]
then
  export OMP_NUM_THREADS=4;
  flags="--no_cuda"
else
  flags=""
fi
PYTORCH_PRETRAINED_BERT_CACHE=caches/bert-base-uncased PYTHONPATH=. NO_ETAG=1 python3 examples/run_squad.py --bert_model bert-base-uncased --load_checkpoint "${model_file}" --run_server --do_lower_case --max_seq_length 384 --doc_stride 128 ${flags} --output_dir /tmp/bert_squad_server_out
