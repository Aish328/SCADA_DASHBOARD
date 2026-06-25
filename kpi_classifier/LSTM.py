import torch
print(torch.__version__)
from dataload import *
from model import *
import torch.nn as nn 

class LSTMModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100,num_layers=2,  batch_first=True)
        self.fc= nn.Linear(100,1)

    def forward(self,x):
        out, _ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out
model = LSTMModel()

