import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
  def __init__(self, d_model, num_heads):
    __super__.init()
    self.d_model = d_model
    self.d_key = d_model // num_heads
    
    self.W_q = nn.Linear(self.d_model, self.d_model)
    self.W_k = nn.Linear(self.d_model, self.d_model)
    self.W_v = nn.Linear(self.d_model, self.d_model)
    self.W_o = nn.Linear(self.d_model, self.d_model)
    
    
    
  def forward(self, x):
    Q = self.W_q(x)
    K = self.W_k(x)
    V = self.W_v(x) 

    # transform into BS x seq_len x num_heads x d_key =>  BS x num_heads x seq_len x d_key
    Q = Q.reshape(Q.shape[0], Q.shape[1], -1, self.d_key).permute(0,2,1,3)
    K = K.reshape(K.shape[0], K.shape[1], -1, self.d_key).permute(0,2,1,3)
    V = V.reshape(V.shape[0], V.shape[1], -1, self.d_key).permute(0,2,1,3)

    attn_logits = Q @ K.transpose(-2,-1)/math.sqrt(self.d_key) # seq_len * seq_len
    attn_weights = F.softmax(attn_logits, dim = -1) 
    context_vector = attn_weights @ V # V has trailing d_key
    concatted = context_vector.contiguous().transpose(1,2).reshape(Q.shape[0], Q.shape[1], -1) 
    return self.W_o(concatted)
