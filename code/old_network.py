import re
from tokenize import group
import torch.nn as nn
import torch.nn.functional as F 
import torch
import numpy as np
from torch.nn.modules.conv import ConvTranspose2d
import model
import util
from einops import rearrange, reduce, repeat
import math
import pdb
from functools import lru_cache

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

###########################################################
## From VRT: https://github.com/JingyunLiang/VRT
class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features # 4*32*i
        hidden_features = hidden_features or in_features # 7*32*i
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat*4, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False, groups = 4),
                                  nn.PixelUnshuffle(2)) # (b (n c//2) h w) -> (b (n 2c) h//2 w//2)

    def forward(self, x):
        B,_,C,H,W = x.shape
        x = rearrange(x, 'b n c h w -> b (n c) h w')
        x = self.body(x) # exposure-wise sampling: (b*4, h/4, w/4, 32) 
        x = rearrange(x, 'b (n c) h w -> b n c h w', n=4)
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat*4, n_feat*8, kernel_size=3, stride=1, padding=1, bias=False, groups = 4),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        x = rearrange(x, 'b n c h w -> b (n c) h w')
        x = self.body(x) # exposure-wise sampling: (b*4, h/4, w/4, 16) 
        x = rearrange(x, 'b (n c) h w -> b n c h w', n=4)
        return x

##########################################################################

@lru_cache()
def compute_mask(H, W, window_size, shift_size, device):
    """_summary_"""

    img_mask = torch.zeros((1, H, W, 1), device=device)
    cnt = 0
    for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = img_mask.view(-1, H//window_size[0], window_size[0], W//window_size[1], window_size[1])
    mask_windows = mask_windows.permute(0, 1, 3, 2, 4).contiguous().view(-1, (window_size[0]*window_size[1]))
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    #print(attn_mask[0, ...])
    #print(attn_mask.shape)
    return attn_mask

class TransformerBlock(nn.Module):
    """ Perform exposure alignment and self attention then combine.

    Args:
        num_heads (int): # of attention head.
        window_size (tuple[int]): Spatial window size with shape of (w, c).
        layer_size (tuple[int]): Dimensions for the LayerNorm with shape of (exp_channels, H, W).
        exp_channels (int): # of input channels of each exposure.
    """
    # TODO: change variables for better understanding
    def __init__(self, num_heads, window_size, layer_size):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.exp_channels = layer_size[0]
        self.layer_size = layer_size
        self.scale = (layer_size[0])**-0.5
        
        self.layerNorm = nn.LayerNorm(layer_size[0]) # TODO: need change to N*exp_channels
        self.layerNorm2 = nn.LayerNorm(layer_size[0]*4) # TODO: need change to N*exp_channels
        self.proj_ea = nn.Linear(layer_size[0],layer_size[0]*num_heads*3, bias = False)
        self.proj_sa = nn.Linear(layer_size[0],layer_size[0]*num_heads*3, bias = False)
        self.register_buffer("position_bias",
                             self.get_sine_position_encoding(window_size[1:], layer_size[0]//2 , normalize=True))
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = Mlp_GEGLU(in_features=layer_size[0]*12*self.num_heads, hidden_features = layer_size[0]*self.num_heads,out_features=4*layer_size[0], act_layer=nn.GELU)
        # self.mlp = Mlp_GEGLU(in_features=layer_size[0]*8, hidden_features = layer_size[0]*4, out_features=4*layer_size[0], act_layer=nn.GELU)
        self.mlp2 = Mlp_GEGLU(in_features=layer_size[0]*4, hidden_features = layer_size[0]*4,out_features=layer_size[0]*4, act_layer=nn.GELU)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (B, nE (4), C, H, W).
            mask: (0/-inf) mask for padding and shifted window method.
        """
        B,_,C,H,W = x.shape
        x_origin = torch.clone(x)

        # TODO: padding for various resolution
        # pad feature maps to multiples of window size
        window_size = self.window_size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = 0
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, 0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b), mode='constant')
        _,_,_,Hp,Wp = x.shape

        if mask is not None:
            x = torch.roll(x, shifts=(-4, -4), dims=(3, 4))
            attn_mask = mask
        else:
            attn_mask = None
        
        x = x.view(B, 4, self.exp_channels, Hp//self.window_size[0], self.window_size[0],
                   Wp//self.window_size[1], self.window_size[1])
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(-1, (self.window_size[0]*self.window_size[1]), self.exp_channels)
        x = self.layerNorm(x) 
        B_, W_, C_ = x.shape # x: (B*4*nW, W, c)
        
        # Perform exposure alignment and self-attention
        # qkv_ea: (4, 3, B_//4, W_, C_)
        # For EA => q_i, k_i, v_i: (B nW) nH W C (i: 1~4)
        # For SA => q, k, v: (B nW) nH W (E C)
        qkv_ea = rearrange(self.proj_ea(x + self.position_bias.repeat(1, 1, 1)), '(B E nW) W (nP nH C) -> E nP (B nW) nH W C', B=B, E=4, nH=self.num_heads, nP=3)
        qkv_sa = rearrange(self.proj_sa(x), '(B E nW) W (nP nH C) -> nP (B nW) nH W (E C)', B=B, E=4, nH=self.num_heads, nP=3)
        
        exposure_att1 = []
        exposure_att2 = []
        for exp_level in range(4):
            exposure_att1.append(self.attention(qkv_ea[0,0], qkv_ea[exp_level,1], qkv_ea[exp_level,2], (B_, W_, C_)))
            exposure_att2.append(self.attention(qkv_ea[exp_level, 0], qkv_ea[0,1], qkv_ea[0,2], (B_, W_, C_)))
        exposure_att1 = torch.cat(exposure_att1, dim = -1) 
        exposure_att2 = torch.cat(exposure_att2, dim = -1)
        exposure_att1 = rearrange(exposure_att1, 'B nH W C -> B W (nH C)', nH = self.num_heads)
        exposure_att2 = rearrange(exposure_att2, 'B nH W C -> B W (nH C)', nH = self.num_heads)

        q, k, v = qkv_sa[0], qkv_sa[1], qkv_sa[2] 
        self_att = self.attention(q, k, v, (B_, W_, C_), True)
        self_att = rearrange(self_att,'B nH W C -> B W (nH C)', nH = self.num_heads)
        
        if False:
            x = torch.cat((exposure_att1, exposure_att2), dim=-1) # : (B*nW, W, 2*4c)
            x = self.mlp(x) # : (B*nW, W, 4c)
        else:
            x = torch.cat((self_att, exposure_att1, exposure_att2), dim=-1) # : (B*nW, W, 3*4c)
            x = self.mlp(x) # : (B*nW, W, 4c)

        x = x.view(B, H//self.window_size[0], W//self.window_size[1], 
                   self.window_size[0], self.window_size[1], 4, self.num_heads, self.exp_channels//self.num_heads) # nW -> nH * nW (width)
        x = x.permute(0, 5, 6, 7, 3, 1, 4, 2).contiguous().view(B, 4, self.exp_channels, Hp, Wp) 

        if mask is not None:
            x = torch.roll(x, shifts=(4, 4), dims=(3, 4))

        # End padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :, :H, :W]

        x = x + x_origin
        x_origin = torch.clone(x)
        x = rearrange(x, 'B N C H W -> B H W (N C)')
        x = self.layerNorm2(x)
        x = self.mlp2(x)
        x = rearrange(x, 'B H W (N C) -> B N C H W', N = 4)
        x = x_origin + x
        return x

    def attention(self, q, k, v, x_shape, relative_position_encoding=False, mask=None):
        """ Attention operation.

        Args:
            q, k, v: query, key and value with shape of (B*nH, nW, C).
            x_shape (tuple[int]): shape of result vector.
            relative_position_encoding (bool): bias for self-attention.
        """
        B_, N, C = x_shape
        att_mat = (q * self.scale) @ k.transpose(-2, -1) # : (B*nW, W, W)

        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
            att_mat = att_mat + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N
        
        if mask is None:
            att_weight = self.softmax(att_mat)
        else:
            nW = mask.shape[0]
            att_weight = att_weight.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            att_weight = att_weight.view(-1, self.num_heads, N, N)
            att_weight = self.softmax(att_weight)
        
        return att_weight @ v

    # From: SWIN Transformer implementation
    def get_position_index(self, window_size):
        """Get pair-wise relative position index for each token inside the window.

        Args:
            window_size (int): _description_

        Returns:
            _type_: _description_
        """

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    # From: VRT implementation
    def get_sine_position_encoding(self, HW, num_pos_feats=8*8, temperature=10000, normalize=False, scale=None):
        '''Get sine position encoding'''

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()
    
class SRHDR(nn.Module):
    """ Proposed Network.
    
    Args:
        args: 

    """
    def __init__(self, args, conv=util.default_conv):
        super(SRHDR, self).__init__()

        self.is_hdr = args.is_hdr
        self.is_origin_size = args.is_origin_size

        # For the ablation study
        self.demosaicing = args.demosaicing
        self.multiscale = args.multiscale
        self.hdrfusion = args.hdrfusion
        self.transformer = args.transformer

        self.f1_num = 24
        self.input_size = args.input_size

        # Demosaicing
        self.first_conv = nn.Conv2d(in_channels = 4, out_channels = self.f1_num, kernel_size = (3,3), padding = 'same')
        if self.demosaicing:
            self.first_deconv = nn.ConvTranspose2d(self.f1_num, self.f1_num*2, kernel_size = (3,3), padding = 1)
            sr1_feats = self.f1_num*2
            sr1_kernel_size = 3
            sr1_reduction = 16
            sr1_n_resblocks = 3 
            sr1_n_resgroups = 2 
            modules_sr1 = [
                ResidualGroup(
                    conv, sr1_feats, sr1_kernel_size, sr1_reduction, sr1_n_resblocks) \
                for _ in range(sr1_n_resgroups)]
            self.sr1 = nn.Sequential(*modules_sr1)
            self.sr_deconv = nn.ConvTranspose2d(sr1_feats, sr1_feats, kernel_size = (3,3), stride = 2, padding = 1, output_padding=1)
            self.sr_conv = nn.Conv2d(sr1_feats, out_channels = self.f1_num, kernel_size = (3,3), padding = 'same')
        else:
            self.sr_deconv = nn.ConvTranspose2d(self.f1_num, self.f1_num, kernel_size = (3,3), stride = 2, padding = 1, output_padding=1)

        # Transformer part
        if self.transformer:
            self.transformerBlock11 = TransformerBlock(2, (8,8,8), (self.f1_num, self.input_size//2, self.input_size//2)) # first block, first scale
            self.transformerBlock11_shift = TransformerBlock(2, (8,8,8), (self.f1_num, self.input_size//2, self.input_size//2))
            self.transformerBlock21 = TransformerBlock(2, (8,8,8), (self.f1_num, self.input_size//2, self.input_size//2))
            self.transformerBlock21_shift = TransformerBlock(2, (8,8,8), (self.f1_num, self.input_size//2, self.input_size//2))

            if self.multiscale:
                self.transformerBlock12 = TransformerBlock(2, (8,8,8), (self.f1_num*2, self.input_size//4, self.input_size//4))
                self.transformerBlock12_shift = TransformerBlock(2, (8,8,8), (self.f1_num*2, self.input_size//4, self.input_size//4))
                self.transformerBlock13 = TransformerBlock(2, (8,8,8), (self.f1_num*4, self.input_size//8, self.input_size//8))
                self.transformerBlock13_shift = TransformerBlock(2, (8,8,8), (self.f1_num*4, self.input_size//8, self.input_size//8))
                self.transformerBlock22 = TransformerBlock(2, (8,8,8), (self.f1_num*2, self.input_size//4, self.input_size//4))
                self.transformerBlock22_shift = TransformerBlock(2, (8,8,8), (self.f1_num*2, self.input_size//4, self.input_size//4))
                
                self.down1 = Downsample(self.f1_num)
                self.down2 = Downsample(self.f1_num * 2)
                self.up2 = Upsample(self.f1_num * 4)
                self.up1 = Upsample(self.f1_num * 2)

            self.final_conv = nn.Conv2d(in_channels = self.f1_num*4, out_channels = self.f1_num*4, kernel_size = (3,3), padding = 'same', groups=4)
            
        # Reconstruction
        if self.is_hdr:
            if self.hdrfusion:
                self.hdr_fusion = util.SpatialGate(self.f1_num*4)
            self.upscale = nn.Sequential(nn.Conv2d(in_channels = self.f1_num*4, out_channels = self.f1_num*4, kernel_size = (3,3), padding = 'same'),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels = self.f1_num*4, out_channels = self.f1_num*4, kernel_size = (3,3), padding = 'same'),
                                     nn.PixelShuffle(2),
                                     nn.Conv2d(in_channels = self.f1_num, out_channels = 3, kernel_size = (3,3), padding = 'same'))
      
        elif self.is_origin_size:
            self.origin_size_conv = nn.Sequential(nn.Conv2d(in_channels = self.f1_num*4, out_channels = self.f1_num, kernel_size = (3,3), padding = 'same', groups=4),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels = self.f1_num, out_channels = 12, kernel_size = (3,3), padding = 'same', groups=4))
        else:
            self.upscale = nn.Sequential(nn.Conv2d(in_channels = self.f1_num*4, out_channels = self.f1_num*4, kernel_size = (3,3), padding = 'same', groups=4),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels = self.f1_num*4, out_channels = self.f1_num*4, kernel_size = (3,3), padding = 'same', groups=4),
                                     nn.PixelShuffle(2),
                                     nn.Conv2d(in_channels = self.f1_num, out_channels = 12, kernel_size = (3,3), padding = 'same', groups=4))
        
    def forward(self, x):
        b,h,w = x.shape # x: (B, H, W)

        # Channel-wise and exposure-wise subsampling: (b,h,w) -> (b,4,4,h/4,w/4)
        sub_x = torch.zeros(b,4,4,h//4,w//4).to(torch.device('cuda'))
        for level in range(4):
            offset_h = level//2
            offset_w = level%2
            sub_x[:,level,0,:,:] = x[:,offset_h::4,offset_w::4]
            sub_x[:,level,1,:,:] = x[:,offset_h::4,2+offset_w::4]
            sub_x[:,level,2,:,:] = x[:,2+offset_h::4,offset_w::4]
            sub_x[:,level,3,:,:] = x[:,2+offset_h::4,2+offset_w::4]

        # Demosaicing
        x = rearrange(sub_x, 'B E C H W->(B E) C H W', E = 4, C = 4)
        x = self.first_conv(x)
        if self.demosaicing:
            x = self.first_deconv(x)
            sr_x = self.sr1(x)
            x = sr_x + x
            x = self.sr_deconv(x)
            x = self.sr_conv(x)
        else:
            x = self.sr_deconv(x)
        x = rearrange(x, '(B E) C H W -> B E C H W', E = 4)
        _, _, _, xH, xW = x.shape

        attn_mask = compute_mask(xH, xW, window_size=(8,8), shift_size=(4,4), device=x.device)
        
        ####### Featrue Alignment #######
        if self.transformer:
            x_scale11 = self.transformerBlock11(x) #scale 1 // (b, 4*32, h/2, w/2)
            x_scale11 = self.transformerBlock11_shift(x_scale11, attn_mask)
            if self.multiscale:
                x_scale12 = self.down1(x_scale11) # exposure-wise sampling: (b*4, h/4, w/4, 64) 
                x_scale12_out = self.transformerBlock12(x_scale12)
                x_scale12_out = self.transformerBlock12_shift(x_scale12_out, attn_mask)
                
                x_scale13 = self.down2(x_scale12_out) # : (b, 4*64, h/8, w/8)
                x_scale13_out = self.transformerBlock13(x_scale13)
                x_scale13_out = self.transformerBlock13_shift(x_scale13_out, attn_mask)
                
                x_scale23 = self.up2(x_scale13_out) # : (b, 4*32, h/4, w/4)
            
                x_scale22 = self.transformerBlock22(x_scale23 + x_scale12) # addition or (concat and 1x1conv)
                x_scale22 = self.transformerBlock22_shift(x_scale22, attn_mask)
                x_scale22_out = self.up1(x_scale22) # : (b, 4*16, h/2, w/2)
                
                x_scale21 = self.transformerBlock21(x_scale22_out + x_scale11)
                x_scale21 = self.transformerBlock21_shift(x_scale21, attn_mask)
            else:
                x_scale21 = self.transformerBlock21(x_scale11)
                x_scale21 = self.transformerBlock21_shift(x_scale21, attn_mask)

            x_scale21 = rearrange(x_scale21, 'B N C H W -> B (N C) H W')
            x_final = self.final_conv(x_scale21)
            x_final = rearrange(x_final, 'B (N C) H W -> B N C H W', N=4)
            x = x + x_final # (b, 4, 32, h/2, w/2)

        if self.is_hdr:
            ####### HDR fusion & Upscaling #######
            x = rearrange(x, 'B N C H W -> B (N C) H W')
            if self.hdrfusion:
                x_extra = self.hdr_fusion(x)
                x = x + x_extra
            x = self.upscale(x)

        elif self.is_origin_size:
            # Convert to RGB images: (b, 4, 32, h/2, w/2) -> (b, 4, 3, h//2, w//2)
            x = rearrange(x, 'B N C H W -> B (N C) H W')#(B N) H W C')
            x = self.origin_size_conv(x)
            x = rearrange(x, 'B (N C) H W -> B N C H W', C = 3)

        else:
            # Upscale: (b, 4, 32, h/2, w/2) -> (b, 4, 3, h, w)
            x = rearrange(x, 'B N C H W -> B (N C) H W')
            x = self.upscale(x)
            x = rearrange(x, 'B (N C) H W-> B N C H W',C=3)
        
        return x