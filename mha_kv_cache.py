import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    assert d_model % num_heads == 0, "d_model % num_heads != 0"
    self.d_key = d_model // num_heads

    self.W_k = nn.Linear(self.d_model, self.d_model)
    self.W_q = nn.Linear(self.d_model, self.d_model)
    self.W_v = nn.Linear(self.d_model, self.d_model)
    self.W_o = nn.Linear(self.d_model, self.d_model)
    self.dropout = nn.Dropout() # default dropout percentage

  def forward(self, x):
    BS, T, _ = x
    
    Q = self.W_q(x)
    K = self.W_k(x)
    V = self.W_v(x)

    # MHA: BS X T X H X d_key
    Q = Q.view(BS, T, self.num_heads, -1).transpose(1,2)
    K = K.view(BS, T, self.num_heads, -1).transpose(1,2)
    V = V.view(BS, T, self.num_heads, -1).transpose(1,2)

    attn_logits = Q @ K.transpose(-2,-1)/math.sqrt(self.d_key) # swap last two dimensions
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_weights = self.dropout(attn_weights)
    concatted_heads = attn_weights.transpose(1,2).contiguous().view(BS, T, -1)
    return self.W_o(concatted_heads)
    
    
  
    
    
