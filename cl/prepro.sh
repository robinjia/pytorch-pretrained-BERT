#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bert-params-tf <name>" 1>&2
  exit 1
fi
input="$1"
name="$2"
cl run bert:${input} :bert-code 'python3.6 bert-code/convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path bert/bert_model.ckpt --bert_config_file bert/bert_config.json --pytorch_dump_path pytorch_model.bin' -n ${name} --request-memory 8g
