#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 <base|large> <cased|uncased>" 1>&2
  exit 1
fi
size="$1"
case="$2"
params_tf="bert-${size}-${case}-tf"
params_torch="bert-${size}-${case}-torch"
train_cmd='python3.6 bert-code/run_squad.py --vocab_file bert/vocab.txt --bert_config_file bert/bert_config.json --init_checkpoint bert-torch/pytorch_model.bin --do_train --do_predict --do_lower_case --train_file train-v1.1.json --predict_file dev-v1.1.json --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --save_checkpoints_steps 100000 --output_dir out'
eval_cmd='python evaluate-v1.1.py dev-v1.1.json out/predictions.json > eval.json'
if [ "$size" = "large" ]
then
  train_flags="--train_batch_size 24 --gradient_accumulation_steps 2 --optimize_on_cpu"
  num_gpus=4
else
  train_flags="--train_batch_size 6"
  num_gpus=1
fi
cmd="${train_cmd} ${train_flags}; ${eval_cmd}"
cl run bert:$params_tf :bert-code bert-torch:$params_torch train-v1.1.json:0x7e0a0a21057c4d989aa68da42886ceb9 dev-v1.1.json:0x8f29fe78ffe545128caccab74eb06c57 evaluate-v1.1.py:0xbcd57bee090b421c982906709c8c27e1 "$cmd" --request-gpus ${num_gpus} --request-memory 16g --request-docker-image codalab/default-gpu -n train-${size}-${case} --request-queue tag=bert
