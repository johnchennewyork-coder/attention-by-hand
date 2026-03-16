


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
  def __init__(self, d_model, num_heads, causal_mask=False):
    self.d_model = d_model
    assert d_model % num_heads == 0, "num_heads must cleanly divide d_model"
    self.d_key = d_model//num_heads
    self.num_heads = num_heads
    self.W_qkv = nn.Linear(self.d_model, 3*self.d_model)
    self.W_o = nn.Linear(self.d_model, self.d_model)
    self.causal_mask = causal_mask

def forward(self, x):
  BS, T, _ = x.shape
  QKV = self.W_qkv(x)
  Q, K, V  = QKV.split(self.d_model, dim=-1) # split on the last dimension

  # MHA 
  Q = Q.view(BS, T, self.num_heads, self.d_key).transpose(1,2)
  K = K.view(BS, T, self.num_heads, self.d_key).transpose(1,2)
  V = V.view(BS, T, self.num_heads, self.d_key).transpose(1,2)

  # K = BS X MHA X T X D_KEY
  attn_logits = Q @ K.transpose(-1, -2)/math.sqrt(self.d_key)
  attn_weights = F.softmax(attn_logits, dim=-1)
  context_vector = attn_weights @ V # BS MHA T_q, T_k
  # reshape, self.d_key by self.d_key 
  concatted_heads = context_vector.transpose(1,2).contiguous().reshape(BS, T, self.d_model)
  return self.W_o(concatted_heads)
  
