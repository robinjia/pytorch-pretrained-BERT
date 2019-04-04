#!/bin/bash
set -eu -o pipefail
if [ $# == 0 ]
then
  echo "Usage: $0 [base|large| [cased|uncased| out-dir train.json <num epochs (usually 2)>"
  exit 1
fi
SIZE="$1"
CASING="$2"
OUT_DIR="$3"
TRAIN_FILE="$4"
EPOCHS="$5"  # Usually do 2
mkdir -p $OUT_DIR
SQUAD_DIR=squad_data
CACHE_DIR=bert_cache
if [ "$SIZE" == "large" ]
then
  echo "Note: requires 4 GPUs"
  batch_size=24
  #batch_size=16
else
  #batch_size=32  # Not sure what correct batch size is
  batch_size=16  # Not sure what correct batch size is
fi
PYTHONPATH=. PYTORCH_PRETRAINED_BERT_CACHE=$CACHE_DIR python examples/run_squad2.py \
  --bert_model bert-${SIZE}-${CASING} \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file=$TRAIN_FILE \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --learning_rate=3e-5 \
  --num_train_epochs=$EPOCHS \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUT_DIR \
  --train_batch_size=$batch_size \
  --predict_batch_size=$batch_size \
  --fp16 \
  --loss_scale 128
