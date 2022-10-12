import torch.nn as nn
import torch.nn.functional as F 
import torch
import numpy as np
from torch.nn.modules.conv import ConvTranspose2d
import model
import util

def make_model(args, parent=False):
    return SRHDR(args)

#############From AHDR####################: https://github.com/qingsenyangit/AHDRNet
class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
        
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

#############From RCAN####################: https://github.com/yulunzhang/RCAN
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Proposed Network
class SRHDR(nn.Module):
    def __init__(self, args, conv=util.default_conv):
        super(SRHDR, self).__init__()

        # Demosaicing & denoising
        self.f1_num = 64

        self.first_conv = nn.Conv2d(in_channels = 4, out_channels = self.f1_num, kernel_size = (3,3), padding = 'same')
        self.deconv = nn.ConvTranspose2d(self.f1_num, 64, kernel_size = (3,3), stride = 2, 
            padding = 1, output_padding=1)

        sr1_feats = 64
        sr1_kernel_size = 3
        sr1_reduction = 16
        sr1_n_resblocks = 4 
        sr1_n_resgroups = 2 
        modules_sr1 = [
            ResidualGroup(
                conv, sr1_feats, sr1_kernel_size, sr1_reduction, sr1_n_resblocks) \
            for _ in range(sr1_n_resgroups)]
        self.sr1 = nn.Sequential(*modules_sr1)

        self.redeconv = nn.ConvTranspose2d(64, 64, kernel_size = (3,3), padding = 1)
        self.reconv = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (3,3), padding = 'same')

        # Edge extraction: https://github.com/yun-liu/RCF-PyTorch
        edge_input_channel = 32 + 4
        self.conv1_1 = nn.Conv2d( edge_input_channel,  64, 3, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d( 64,  64, 3, padding=1, dilation=1)
        self.conv2_1 = nn.Conv2d( 64, 128, 3, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.act = nn.ReLU(inplace=True)

        self.conv1_1_down = nn.Conv2d( 64, 21, 1)
        self.conv1_2_down = nn.Conv2d( 64, 21, 1)
        self.conv2_1_down = nn.Conv2d(128, 21, 1)
        self.conv2_2_down = nn.Conv2d(128, 21, 1)
        self.conv3_1_down = nn.Conv2d(256, 21, 1)
        self.conv3_2_down = nn.Conv2d(256, 21, 1)
        self.conv3_3_down = nn.Conv2d(256, 21, 1)

        self.score_dsn1 = nn.Conv2d(21, 16, 1)
        self.score_dsn2 = nn.Conv2d(21, 16, 1)
        self.score_dsn3 = nn.Conv2d(21, 16, 1)

        self.deconv_stage1 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.deconv_stage2 = nn.ConvTranspose2d(16, 16, 4, stride=4)
        self.add_conv = nn.Conv2d(48, 36, 1)

        # Attention-based HDR fusion: https://github.com/qingsenyangit/AHDRNet
        nChannel = 36
        nFeat = 64
        nDenselayer = 4
        growthRate = 32

        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)

        # CAM
        self.channel_att = util.CAM_Module(nChannel)
        self.conv_cam = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True)

        # SR reconstruction 
        sr2_feats = 128
        sr2_kernel_size = 3
        sr2_reduction = 16
        sr2_n_resblocks = 6
        sr2_n_resgroups = 2 

        modules_sr2 = [
            ResidualGroup(
                conv, sr2_feats, sr2_kernel_size, sr2_reduction, sr2_n_resblocks) \
            for _ in range(sr2_n_resgroups)]

        modules_upsample = [
            util.Upsampler(conv, sr2_feats, act=False),
            conv(sr2_feats, 3, sr2_kernel_size)]

        self.sr2 = nn.Sequential(*modules_sr2)
        self.upsample = nn.Sequential(*modules_upsample)

    # Input x: [# of batch, h, w]
    def forward(self, x):
        b,h,w = x.shape

        # Subsampling: (b,h,w) -> (b,4,h/2,w/2)
        sub_x = torch.zeros(b,4,h//2,w//2).to(torch.device('cuda'))
        for level in range(4):
            offset_h = level//2
            offset_w = level%2
            sub_x[:,level,::2,::2] = x[:,offset_h::4,offset_w::4]
            sub_x[:,level,1::2,::2] = x[:,2+offset_h::4,offset_w::4]
            sub_x[:,level,::2,1::2] = x[:,offset_h::4,2+offset_w::4]
            sub_x[:,level,1::2,1::2] = x[:,2+offset_h::4,2+offset_w::4]

        # Subsampling: (b,h,w) -> (b,4,4,h/4,w/4)
        samp_x = torch.zeros(b,4,4,h//4,w//4).to(torch.device('cuda'))
        for level in range(4):
            offset_h = level//2
            offset_w = level%2
            samp_x[:,level,0,:,:] = x[:,offset_h::4,offset_w::4]
            samp_x[:,level,1,:,:] = x[:,offset_h::4,2+offset_w::4]
            samp_x[:,level,2,:,:] = x[:,2+offset_h::4,offset_w::4]
            samp_x[:,level,3,:,:] = x[:,2+offset_h::4,2+offset_w::4]

        B,E,C,H,W = samp_x.shape # Reshape for batching: (b,4,4,h/4,w/4) -> (-1,4,h/4,w/4)
        x = samp_x.view(-1,C,H,W)
        x = self.first_conv(x) # (-1,4,h/4,w/4) -> (-1,128,h/4,w/4)
        x = self.deconv(x) # (-1,128,h/4,w/4) -> (-1,64,h/4,w/4)

        # Demosaicing: 
        sr_x = self.sr1(x)
        x = sr_x + x
        x = self.redeconv(x)
        x = self.reconv(x)
        x = x.view(B,E,-1,2*H,2*W) # (2, 4, 6, 128, 128)

        # Skip connection of raw input
        sub_x = sub_x.unsqueeze(1).repeat(1,E,1,1,1)
        x = torch.cat((x,sub_x),2)
        _,_,ch,_,_ = x.shape
        x = x.view(-1,ch,2*H,2*W)
        # TODO: Add edge module
        # Edge extraction
        conv1_1 = self.act(self.conv1_1(x))
        conv1_2 = self.act(self.conv1_2(conv1_1))
        pool1   = self.pool1(conv1_2)
        conv2_1 = self.act(self.conv2_1(pool1))
        conv2_2 = self.act(self.conv2_2(conv2_1))
        pool2   = self.pool2(conv2_2)
        conv3_1 = self.act(self.conv3_1(pool2))
        conv3_2 = self.act(self.conv3_2(conv3_1))
        conv3_3 = self.act(self.conv3_3(conv3_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        
        out1 = self.score_dsn1(conv1_1_down + conv1_2_down)
        out2 = self.score_dsn2(conv2_1_down + conv2_2_down)
        out3 = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        
        out2 = self.deconv_stage1(out2)
        out3 = self.deconv_stage2(out3)
        fusion_out = torch.cat((out1, out2, out3), dim=1)
        fusion_out = self.add_conv(fusion_out)
        x = fusion_out + x
        x = x.view(B,E,-1,2*H,2*W)

        # HDR fusion
        x1 = x[:,0,:,:,:]
        x2 = x[:,1,:,:,:]
        x3 = x[:,2,:,:,:]
        x4 = x[:,3,:,:,:]

        F1_ = self.relu(self.conv1(x1))
        F2_ = self.relu(self.conv1(x2))
        F3_ = self.relu(self.conv1(x3))
        F4_ = self.relu(self.conv1(x4))

        F = torch.cat((F1_,F2_,F3_,F4_),dim=1)
        F = self.channel_att(F)
        FDF = self.conv_cam(F)

        # Super-Resolution: Use RCAB
        res = self.sr2(FDF)
        res = self.upsample(res)
        # Result shape: (B,3,H,W)

        return res