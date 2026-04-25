

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    self.i2h = nn.Linear(input_size, hidden_size)
    self.h2o = nn.LSTM(hidden_size, output_size)

  def forward(self, x):
    intermediate = self.i2h(x)
    hidden, cell_tape = self.h2o(intermediate)
    return hidden

