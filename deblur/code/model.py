import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable
import network
import torchsummary

class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.device = device
        print('Making model...')
        self.model = network.make_model(args).to(self.device)
        
    def forward(self, x):
        return self.model(x)
    
    