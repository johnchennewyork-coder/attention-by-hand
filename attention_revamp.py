

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
  def __init__(self, d_model, num_heads, causal_mask=False):
    super().__init__()
    self.d_model = d_model
    assert  d_model % num_heads == 0, "num_heads must divide d_model cleanly"
    self.d_key = d_model // num_heads
    self.num_heads = num_heads
    self.W_qkv = nn.Linear(d_model, 3*d_model)
    self.W_o = nn.Linear(d_model,d_model)
    self.dropout = nn.Dropout(p=0.2)
    
  def forward(self, x):
    BS, T_q, d_model = x.shape

    QKV = self.W_qkv(x)
    Q, K, V = QKV.split(self.d_model, dim=2)

    # transpose, reshaping
    mha_q = Q.reshape(BS, T_q, self.num_heads, self.d_key).transpose(1,2)
    mha_k = K.reshape(BS, T_q, self.num_heads, self.d_key).transpose(1,2)
    mha_v = V.reshape(BS, T_q, self.num_heads, self.d_key).transpose(1,2)

    attn_logits = mha_q @ mha_k.transpose(-2,-1)/math.sqrt(self.d_key) 
    attn_weights = F.softmax(attn_logits, dim=-1)
    context_vector = attn_weights @ mha_v # 
    concatted_vec = context_vector.transpose(1,2).contiguous().view(BS, T_q, self.d_model)
    output_vec  = self.W_o(concatted_vec)
    return output_vec 
    
