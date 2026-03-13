import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# torch.nn.functional.scaled_dot_product_attention (SDPA)
class Attention(nn.Module):
  def __init__(self, d_model, num_heads, dropout_p=0.1, causal_mask=False):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    assert d_model % num_heads == 0, "d_model % num_heads != 0"
    self.d_key = d_model // num_heads

    self.W_k = nn.Linear(self.d_model, self.d_model)
    self.W_q = nn.Linear(self.d_model, self.d_model)
    self.W_v = nn.Linear(self.d_model, self.d_model)
    self.W_o = nn.Linear(self.d_model, self.d_model)
    self.dropout = nn.Dropout(dropout_p) # default dropout percentage
    self.causal_mask = causal_mask

  def forward(self, x):
    BS, T, _ = x.shape
    
    Q = self.W_q(x)
    K = self.W_k(x)
    V = self.W_v(x)

    # MHA: BS X T X H X d_key
    Q = Q.view(BS, T, self.num_heads, -1).transpose(1,2)
    K = K.view(BS, T, self.num_heads, -1).transpose(1,2)
    V = V.view(BS, T, self.num_heads, -1).transpose(1,2)

    attn_logits = Q @ K.transpose(-2,-1)/math.sqrt(self.d_key) # swap last two dimensions
    if self.causal_mask:
        mask = torch.tril(torch.ones(attn_logits.shape, x.device))  # zero out the upper right 
        attn_logits.masked_fill_(mask == 0, -float('inf'))
      
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_weights = self.dropout(attn_weights)
    context_vec = attn_weights @ V 
    concatted_heads = context_vec.transpose(1,2).contiguous().view(BS, T, -1)
    return self.W_o(concatted_heads)
    
    
  
    
    
