"""Custom functions for SQuAD model."""
import argparse
import json
from ordered_set import OrderedSet
import random
import sys
import torch
from tqdm import tqdm

def run_test_gradients(model, bert_config, examples, features, device):
  random.seed(0)
  model.eval()
  vocab_size = bert_config.vocab_size
  logsoftmax_fn = torch.nn.LogSoftmax(dim=0)
  plot_data = []
  all_info = []
  for ex, feats in zip(tqdm(examples), features):
    #print(ex)
    input_ids = torch.tensor([feats.input_ids], dtype=torch.long, device=device)
    input_mask = torch.tensor([feats.input_mask], dtype=torch.long, device=device)
    segment_ids = torch.tensor([feats.segment_ids], dtype=torch.long, device=device)
    with torch.enable_grad():
      model.zero_grad()
      batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
      start_log_probs = logsoftmax_fn(batch_start_logits.squeeze(0))
      end_log_probs = logsoftmax_fn(batch_end_logits.squeeze(0))
      max_start, start_ind = torch.max(start_log_probs, 0)
      max_end, end_ind = torch.max(end_log_probs, 0)
      max_prob = torch.exp(max_start + max_end)
      max_prob.backward()
      emb_grad = model.bert.embeddings.word_embeddings.weight.grad
      emb_grad_norms = torch.norm(emb_grad, p=2, dim=1)  # L
      max_grad_id = torch.argmax(emb_grad_norms)
    used_ids = OrderedSet(feats.input_ids)
    with torch.no_grad():
      #for t in tqdm(range(1000)):
        #old_id = random.sample(used_ids, 1)[0]
        #new_id = random.sample(range(vocab_size), 1)[0]
      old_id = max_grad_id
      #for new_id in tqdm(range(vocab_size)):
      for new_id in random.sample(range(vocab_size), 10):
        # Estimate the new probability using gradient
        old_vec = model.bert.embeddings.word_embeddings(
            torch.tensor(old_id, dtype=torch.long))
        new_vec = model.bert.embeddings.word_embeddings(
            torch.tensor(new_id, dtype=torch.long))
        cur_grad = emb_grad[old_id]
        estim_diff = torch.dot(emb_grad[old_id], new_vec - old_vec)
        estim_prob = max_prob + estim_diff
        # Compute the actual probability
        new_input_id_list = [new_id if i == old_id else i for i in feats.input_ids]
        new_input_ids = torch.tensor([new_input_id_list], dtype=torch.long,
                                     device=device)
        bsl, esl = model(new_input_ids, segment_ids, input_mask)
        slp, elp = logsoftmax_fn(bsl.squeeze(0)), logsoftmax_fn(esl.squeeze(0))
        new_prob = torch.exp(slp[start_ind] + elp[end_ind])
        new_diff = new_prob - max_prob
        cur_info = {
            'max_prob': max_prob.item(), 
            'new_prob': new_prob.item(),
            'estim_prob': estim_prob.item(), 
            'new_diff': new_diff.item(), 
            'estim_diff': estim_diff.item(),
            'grad_norm': torch.norm(cur_grad, p=2).item(),
            'emb_l2_distance': torch.norm(new_vec - old_vec, p=2).item(),
        }
        all_info.append(cur_info)
    print(json.dumps(all_info))
