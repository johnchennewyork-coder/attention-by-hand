import numpy as np
import math
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
    self.num_heads = num_heads # should we be safe and just keep the # of heads
    

  def forward(self, x):
    Q = self.W_q(x) # d_model by d_model *** BS x seq_len x d_model   where d_model is also the embedding dim
    K = self.W_k(x)
    V = self.W_v(x) 
    if self.num_heads == 1:
      attn_logits = Q @ K.transpose(1,2)/math.sqrt(self.d_key) # BS x seq_len x d_model  *** BS x d_model x seq_len =>  BS x seq_len x seq_len # NOTE this could also be called logits 
      attn_weights = torch.nn.functional.softmax(attn_logits, dim = -1) # normalize across keys; for each query, find the set of keys
      return attn_weights @ V # bs x seq_len x seq_len *** bs x seq_len x d_model => bs x seq_len x d_model (context vector)
     
    # MHA case
    
    # reshape Q, K, V 
    mha_Q = Q.reshape(Q.shape[0], Q.shape[1], -1, self.d_key) #  BS x seq_len x d_model ->  BS x seq_len x num_heads x d_key
    mha_K = K.reshape(K.shape[0], K.shape[1], -1, self.d_key) #  BS x seq_len x d_model ->  BS x seq_len x num_heads x d_key
    mha_V = V.reshape(V.shape[0], V.shape[1], -1, self.d_key) #  BS x seq_len x d_model ->  BS x seq_len x num_heads x d_key

    # BS x num_heads x seq_len x d_key *** BS x num_heads x d_key x seq_len => BS x num_heads x seq_len x seq_len
    mha_attn_logits = Q.permute(0,2,1,3) @ K.permute(0,2,3,1)/math.sqrt(self.d_key)
    mha_attn_weights = torch.nn.functional.softmax(mha_attn_logits, axis = -1)
    # mha_attn_weights @ V # BS x num_heads x seq_len x seq_len *** BS X seq_len X num_heads X self.d_key
    mha_context_vec = mha_attn_weights @ V.permute(0,2,1,3) BS x num_heads x seq_len x seq_len *** BS X num_heads X  seq_len X self.d_key  => BS x num_heads x seq_len x d_key 
    mha_context_vec = mha_context_vec.permute(0,2,1,3) # BS x seq_len x num_heads x d_key
    mha_context_vec = mha_context_vec.reshape(mha_context_vec.shape[0],mha_context_vec.shape[1], -1)
    return self.W_o(mha_context_vec) 
    


  '''
  a note on terminology: QK^T -> scores (alternatively, the logits)
  softmax(scores) -> weights 
  weights * values -> context vector
  
  '''
