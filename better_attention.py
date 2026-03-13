import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
  
  def __init__(self, d_model, num_heads):  
    super().__init__()
    self.d_model = d_model
    self.d_key = self.d_model//num_heads
    
    self.W_q = nn.Linear(self.d_model, self.d_model)
    self.W_k = nn.Linear(self.d_model, self.d_model)
    self.W_v = nn.Linear(self.d_model, self.d_model)
    self.W_o = nn.Linear(self.d_model, self.d_model)
    
  def forward(self, x):
    Q = self.W_q(x)
    K = self.W_k(x)
    V = self.W_v(x) # weight matrix + bias term 
    
    attn_logits = Q @ K.transpose(-2,-1) / math.sqrt(self.d_key)
    attn_weights = F.softmax(attn_logits, dim=-1) 
    context_vector = attn_weight @ V
    
    return context_vector
    
