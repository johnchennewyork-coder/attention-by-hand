

import torch
import torch.nn as nn 
import torch.nn.functional as F

class RNN(nn.Module):
  def __init__(self,input_size, hidden_size, output_size):
    self.input = nn.Linear(input_size, hidden_size)
    self.h1 = nn.Linear(hidden_size, output_size)
    
def forward(self, x):
  intermediate = self.input(x)
  out = nn.ReLu(self.h1(intermediate))
  return out 
    
