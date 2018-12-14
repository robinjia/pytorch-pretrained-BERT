"""Prepare a cache for a specific model (e.g. for codalab)."""
import argparse
import os
import sys

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('bert_model', help='Bert pre-trained model selected in the list: bert-base-uncased, '
                      'bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.')
  parser.add_argument('cache_dir', help='Where to write cache')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  os.environ['NO_ETAG'] = '1'
  BertTokenizer.from_pretrained(OPTS.bert_model, cache_dir=OPTS.cache_dir)
  BertForQuestionAnswering.from_pretrained(OPTS.bert_model, cache_dir=OPTS.cache_dir)

if __name__ == '__main__':
  OPTS = parse_args()
  main()

