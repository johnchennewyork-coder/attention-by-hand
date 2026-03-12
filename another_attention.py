import numpy as np
import torch # can also build an attention module


class Attention(torch.nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()

    self.W_q = torch.nn.Linear(d_model, d_model)
    self.W_k = torch.nn.Linear(d_model, d_model)
    self.W_v = torch.nn.Linear(d_model, d_model)
    self.W_o = torch.nn.Linear(d_model, d_model) # 4 matrices

    assert d_model % num_heads == 0, 'Must have d_model % num_heads == 0'
    self.d_key = d_model // num_heads 

  def forward(self, x):
    Q = self.W_q(x) # d_model by d_model *** BS x seq_len x d_model   where d_model is also the embedding dim
    K = self.W_k(x)
    V = self.W_v(x) 
    attn_logits = Q @ K.transpose(1,2)/math.sqrt(self.d_key) # BS x seq_len x d_model  *** BS x d_model x seq_len =>  BS x seq_len x seq_len # NOTE this could also be called logits 
    attn_weights = torch.nn.Functional.softmax(z, axis = -1) # normalize across keys; for each query, find the set of keys
    return attn_weights @ V # bs x seq_len x seq_len *** bs x seq_len x d_model => bs x seq_len x d_model (context vector)
     


  '''
  a note on terminology: QK^T -> scores (alternatively, the logits)
  softmax(scores) -> weights 
  weights * values -> context vector
  
  '''
