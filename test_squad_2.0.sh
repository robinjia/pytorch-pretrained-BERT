#!/bin/bash
set -eu -o pipefail
if [ $# == 0 ]
then
  echo "Usage: $0 [base|large| [cased|uncased| out-dir thresh"
  exit 1
fi
SIZE="$1"
CASING="$2"
LOAD_DIR="$3"
THRESH="$4"
SQUAD_DIR=squad_data
CACHE_DIR=bert_cache
PYTHONPATH=. PYTORCH_PRETRAINED_BERT_CACHE=$CACHE_DIR python examples/run_squad2.py \
  --bert_model bert-${SIZE}-${CASING} \
  --do_predict \
  --do_lower_case \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --max_seq_length=384 \
  --doc_stride=128 \
  --load_dir=$LOAD_DIR \
  --output_dir=$LOAD_DIR/out_thresh_${THRESH} \
  --fp16 \
  --null_score_diff_threshold=$THRESH
