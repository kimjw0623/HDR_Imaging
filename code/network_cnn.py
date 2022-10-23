import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable
from einops import rearrange

def make_model(args, rank):
    return AHDR(rank)

class Upscale(nn.Module):
    def __init__(self, n_feat):
        super(Upscale, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        x = self.body(x)
        return x

class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(DRDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dilation_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

# Attention Guided HDR, AHDR-Net
class AHDR(nn.Module):
    def __init__(self, rank): # TODO: get numbers from paper
        super(AHDR, self).__init__()
        nChannel = 6
        nDenselayer = 6
        nFeat = 64
        growthRate = 32
        self.rank = rank
        # self.args = args

        self.conv_x1 = nn.Conv2d(8, nFeat, 3, 1, 1)
        self.conv_x2 = nn.Conv2d(16, nFeat, 3, 1, 1)
        self.conv_x3 = nn.Conv2d(8, nFeat, 3, 1, 1)

        self.upscale_x1 = Upscale(nFeat)
        self.upscale_x2 = Upscale(nFeat)
        self.upscale_x3 = Upscale(nFeat)

        # F-1
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        self.att11 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        # self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        # self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # DRDBs 3
        self.RDB1 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = DRDB(nFeat, nDenselayer, growthRate)

        self.RDB3 = DRDB(nFeat, nDenselayer, growthRate)
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv 
        self.conv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

        self.upscale = nn.Sequential(# nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(nFeat, nFeat*4, 3, 1, 1),
                                     nn.PixelShuffle(2),
                                     nn.Conv2d(nFeat, 3, 3, 1, 1))


    def forward(self, x):
        b,h,w,c = x.shape # x: (B H W 2)
        x = rearrange(x, 'B H W C -> B C H W')

        # Channel-wise and exposure-wise subsampling: (b,c,h,w) -> (b,c,4,4,h/4,w/4)
        sub_x = torch.zeros(b,c,4,4,h//4,w//4).cuda(self.rank)#.to(torch.device('cuda'))
        for level in range(4):
            offset_h = level//2
            offset_w = level%2
            sub_x[:,:,level,0,:,:] = x[:,:,offset_h::4,offset_w::4]
            sub_x[:,:,level,1,:,:] = x[:,:,offset_h::4,2+offset_w::4]
            sub_x[:,:,level,2,:,:] = x[:,:,2+offset_h::4,offset_w::4]
            sub_x[:,:,level,3,:,:] = x[:,:,2+offset_h::4,2+offset_w::4]

        sub_x = rearrange(sub_x,'B C E ch H W->B E (C ch) H W')

        x1 = sub_x[:,0,...]  # (B 4 H/4 W/4)
        x3 = sub_x[:,3,...]
        #breakpoint()
        # ForkedPdb().set_trace()
        f1 = self.upscale_x1(self.conv_x1(x1)) # f1: (B 60 H/2 W/2)
        f2 = self.upscale_x2(self.conv_x2(torch.cat((sub_x[:,1,...],sub_x[:,2,...]),dim=1)))
        f3 = self.upscale_x3(self.conv_x3(x3))

        F1_ = self.relu(self.conv1(f1))
        F2_ = self.relu(self.conv1(f2))
        F3_ = self.relu(self.conv1(f3))

        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.relu(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = torch.sigmoid(F1_A)
        F1_ = F1_ * F1_A


        F3_i = torch.cat((F3_, F2_), 1)
        F3_A = self.relu(self.att31(F3_i))
        F3_A = self.att32(F3_A)
        F3_A = torch.sigmoid(F3_A)
        F3_ = F3_ * F3_A

        F_ = torch.cat((F1_, F2_, F3_), 1)

        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        us = self.conv_up(FDF)

        output = self.conv3(us)

        output = self.upscale(output)

        output = torch.sigmoid(output)

        return output