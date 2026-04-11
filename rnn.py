import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

'''
1) basic rnn
2) basic 2 layer rnn
3) pytorch rnn
'''

class UseRNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    self.rnn = nn.RNN(input_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
  
  def forward(self, x):
    rnn_out, hidden = self.rnn(x)
    out = self.out(rnn_out)
    return out, hidden

class RegularNN(nn.Module):
  def __init__(self,input_size, hidden_size, output_size):
    super(RegularNN, self).__init__()
    self.input = nn.Linear(input_size, hidden_size)
    self.h1 = nn.Linear(hidden_size, output_size)
    
  def forward(self, x):
    intermediate = self.input(x)
    out = F.ReLU(self.h1(intermediate))
    return out 

class BasicRNN(nn.Module):
  def __init__(self,input_size, hidden_size, output_size):
    super(BasicRNN, self).__init__()
    self.i2h = nn.Linear(input_size, hidden_size)
    self.h2o = nn.Linear(hidden_size, output_size)
    
  def forward(self, input_tensor, hidden_tensor):
    hidden = self.i2h(x)
    out = self.h2o(F.ReLU(hidden))
    return out 

def generate_train_data():
  vals = torch.randn()
  return vals 

if __name__ == '__main__':
  print('alright')
  
  
