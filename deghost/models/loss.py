#-*- coding:utf-8 -*-  
import math
import torch
import torch.nn as nn
import torchvision
import numpy as np

def range_compressor(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super(L1MuLoss, self).__init__()
        self.mu = mu

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return nn.L1Loss()(mu_pred, mu_label)