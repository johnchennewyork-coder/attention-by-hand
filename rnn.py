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
    super(UseRNN, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
  
  def forward(self, x):
    rnn_out, hidden = self.rnn(x)
    out = self.out(rnn_out)
    return out, hidden

class BasicRNN(nn.Module):
  def __init__(self,input_size, hidden_size, output_size):
    super(BasicRNN, self).__init__()
    self.i2h = nn.Linear(input_size, hidden_size)
    self.h2o = nn.Linear(hidden_size, output_size)
    
  def forward(self, input_tensor, hidden_tensor):
    hidden = self.i2h(x)
    out = self.h2o(F.ReLU(hidden))
    return out 

class Basic2LayerRNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(Basic2LayerRNN, self).__init__()
    self.i2h = nn.RNN(input_size, hidden_size)
    self.h2h = nn.RNN(hidden_size, hidden_size)
    self.h2o = nn.RNN(hidden_size, output_size)

  def forward(self, x):
    rnn_out1, hidden = self.i2h(x)
    rnn_out2, hidden2 = self.h2h(rnn_out1)
    return self.h2o(rnn_out2), hidden2
    
    
  

def generate_train_data():
  vals = torch.randn()
  return vals 

if __name__ == '__main__':
  print('alright')
  
  
