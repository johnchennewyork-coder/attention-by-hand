
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, causal_mask=False, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0, 'num heads must cleanly divide d_model'
        self.d_key = d_model / num_heads
        self.W_qkv = nn.Linear(self.d_model, 3*self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.causal_mask = causal_mask

    def forward(self, x):
        BS, T , _ = x.shape
        QKV = self.W_qkv(x)
        Q, K, V = QKV.split(self.d_model, dim=-1)

        # reshape into MHA format
        Q = Q.view(BS, T, self.num_heads, self.d_key).transpose(1,2)
        K = K.view(BS, T, self.num_heads, self.d_key).transpose(1,2)
        V = V.view(BS, T, self.num_heads, self.d_key).transpose(1,2)

        attn_logits = Q @ K.transpose(2,3)/math.sqrt(self.d_key)
        if self.causal_mask:
          mask = torch.tril(torch.ones(T,T, device = x.device))
          attn_logits.masked_fill_(mask == 0 , -float('inf'))
          
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ V # BS x mha x T_q x d_key
        concatted_heads = context_vector.transpose(1,2).contiguous().reshape(BS, T, self.d_model)
        return self.W_o(concatted_heads)
      
        
        
        

