#!/bin/bash
set -eu -o pipefail
cl up *.py pytorch_pretrained_bert examples -n pytorch-bert-src
