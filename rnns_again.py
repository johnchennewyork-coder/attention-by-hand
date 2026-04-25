

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.i2h = nn.RNN(input_size, hidden_size, batch_first=True) # transformer semantics 
    self.h2o = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    output, hidden = self.i2h(x) # note that RNN returns a sequence of length K 
    out = self.h2o(output)
    return out


'''
BS, T, H (like a transformer). 

Then, output is BS, T, H (like a transformer).  
On the other hand, hidden is:
D, BS, H. (where T is implicitly set to -1) # the 'final' hidden state

recall:
BS , T_q, D_model  => for each example, we produce an output for each timestep , of the same size

What is the input and output of transformer? It returns the same size 
'''
