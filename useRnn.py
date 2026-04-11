

import torch
import torch.nn as nn
import torch.nn.functional as F


class UseRNN(nn.Module):
  def __init__(self,input_size, hidden_size, output_size):
    self.rnn = nn.RNN(input_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
    
  def forward(self, x):
    rnn_out, h = self.rnn(x)
    return self.out(rnn_out), h
    
