

import torch
import torch.nn as nn
import torch.nn.functional as F


class UseRNN(nn.Module):
  def __init__(self,input_size, hidden_size, output_size, mode='many_to_one', batch_first=True):
    super(UseRNN, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, batch_first=batch_first)
    self.out = nn.Linear(hidden_size, output_size)
    self.mode = mode
      
    
    
  def forward(self, x):
    rnn_out, h = self.rnn(x)
    if self.mode == 'many_to_one':
      rnn_out = rnn_out[-1] # last time step
      
    return self.out(rnn_out), h
    
