
import torch.nn as nn
import torch.nn.Functional as F
import math

class MHAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    self.W_qkv = nn.Linear(d_model, 3*d_model)
    self.output = nn.Linear(d_model, d_model) # mixes everything together
    self.d_key = d_model // num_heads
    self.dropout = nn.Dropout()

  def forward(x): # BS x T X d_model -> BS x N x T x d_key  
    bs, T, hidden_dim = x.shape
    QKV = self.W_qkv(x) 
    
    Q, K , V = QKV.split(hidden_dim, dim=-1)
    # transpose all of the dims
    Q = Q.view(bs, T, -1, self.d_key).transpose(1,2) # BS N T D_K
    K = K.view(bs, T, -1, self.d_key).transpose(1,2)
    V = V.view(bs, T, -1, self.d_key).transpose(1,2)
    
    # theres also the scaled dot product attention layer you can immediately import
    attn_logits = Q @ K.transpose(2,3)/math.sqrt(self.d_key)
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_weights = self.dropout(attn_weights)
    context_vector = attn_weights @ V
    concatted = context_vector.transpose(1,2).contiguous().view(bs, T, hidden_dim)
    return self.output(concatted) 
    
