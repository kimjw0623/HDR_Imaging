#-*- coding:utf-8 -*-
import math
import time
import torch
import util
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange, reduce, repeat
from functools import lru_cache
import torch.nn.functional as F 

def make_model(args, rank):
    return HDRTransformer(rank=rank)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class LocalContextExtractor(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)

## From Restormer: https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

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
        out_features = out_features# or in_features # 4*32*i
        hidden_features = hidden_features# or in_features # 7*32*i
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1),
                                  nn.PixelUnshuffle(2)) # (b (n c//2) h w) -> (b (n 2c) h//2 w//2)

    def forward(self, x):
        B,_,C,H,W = x.shape
        x = rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.body(x) # exposure-wise sampling: (b*4, h/4, w/4, 32) 
        x = rearrange(x, '(b n) c h w -> b n c h w', n=3)
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        x = rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.body(x) # exposure-wise sampling: (b*4, h/4, w/4, 16) 
        x = rearrange(x, '(b n) c h w -> b n c h w', n=3)
        return x

@lru_cache()
def compute_mask(H, W, window_size, shift_size, device):
    """_summary_"""

    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_r = (window_size[0] - W % window_size[0]) % window_size[0]
    Hp = H + pad_b
    Wp = W + pad_r

    img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
    cnt = 0
    for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = img_mask.view(-1, Hp//window_size[0], window_size[0], Wp//window_size[1], window_size[1])
    mask_windows = mask_windows.permute(0, 1, 3, 2, 4).contiguous().view(-1, (window_size[0]*window_size[1]))
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
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
    def __init__(self, num_heads, window_size, fnum, idx, keep_query, ffn_dconv, is_shared, is_local):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.exp_channels = fnum
        self.scale = (fnum//num_heads)**-0.5
        self.keep_query = keep_query
        self.ffn_dconv = ffn_dconv
        self.is_shared = is_shared
        self.is_local = is_local

        if idx % 2 == 1:
            self.is_mask = True
        else:
            self.is_mask = False
        
        self.layerNorm = nn.LayerNorm(fnum) # TODO: need change to N*exp_channels
        self.layerNorm2 = nn.LayerNorm(fnum*3) # TODO: need change to N*exp_channels

        self.proj_ea = nn.Linear(fnum,fnum*3, bias=False) # 4: 2 for level 1
        
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("position_bias",
                            self.get_sine_position_encoding(window_size[1:], fnum//2 , normalize=True))
        
        if self.is_shared:
            # self.mlp1 =  Mlp(in_features=fnum*6, hidden_features=fnum*12, out_features=fnum*3, act_layer=nn.GELU)
            self.mlp1 =  Mlp_GEGLU(in_features=fnum*6, hidden_features=fnum*12, out_features=fnum*3, act_layer=nn.GELU)
        else:
            if self.keep_query:
                self.mlp_exp1 = Mlp_GEGLU(in_features=fnum*3, hidden_features=fnum,out_features=fnum, act_layer=nn.GELU)
                self.mlp_exp2 = Mlp_GEGLU(in_features=fnum*3, hidden_features=fnum,out_features=fnum, act_layer=nn.GELU)
                self.mlp_exp3 = Mlp_GEGLU(in_features=fnum*3, hidden_features=fnum,out_features=fnum, act_layer=nn.GELU)
            else:
                self.mlp_exp1 = Mlp_GEGLU(in_features=fnum*2, hidden_features=fnum,out_features=fnum, act_layer=nn.GELU)
                self.mlp_exp2 = Mlp_GEGLU(in_features=fnum*2, hidden_features=fnum,out_features=fnum, act_layer=nn.GELU)
                self.mlp_exp3 = Mlp_GEGLU(in_features=fnum*2, hidden_features=fnum,out_features=fnum, act_layer=nn.GELU)

        if self.ffn_dconv:
            self.conv11 = nn.Conv2d(3*fnum, 3*fnum, kernel_size=1, bias=False)
            self.mlp2 = FeedForward(fnum*3, ffn_expansion_factor=2, bias=False)
        else:
            # self.mlp2 = Mlp(in_features=fnum*3, hidden_features = fnum*12, out_features=fnum*3, act_layer=nn.GELU)
            self.mlp2 = Mlp_GEGLU(in_features=fnum*3, hidden_features = fnum*12, out_features=fnum*3, act_layer=nn.GELU)

        if self.is_local:
            self.lce = LocalContextExtractor(fnum*3)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (B, 3, C, H, W).
            mask: (0/-inf) mask for padding and shifted window method.
        """
        B,_,C,H,W = x.shape
        x_origin = x
        
        # padding
        pad_l = pad_t = 0
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (pad_t, pad_r, pad_l, pad_b))
        _,_,_,Hp,Wp = x.shape

        if self.is_local:
            lcf = rearrange(x, 'B E C H W->B (E C) H W')

        if self.is_mask:
            x = torch.roll(x, shifts=(-4, -4), dims=(3, 4))
            attn_mask = compute_mask(H, W, window_size=(8,8), shift_size=(4,4), device=x.device)
        else:
            attn_mask = None
        
        x = x.view(B, 3, self.exp_channels, Hp//self.window_size[0], self.window_size[0],
                   Wp//self.window_size[1], self.window_size[1])
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous().view(-1, (self.window_size[0]*self.window_size[1]), self.exp_channels)
        
        B_, W_, C_ = x.shape # x: B * 3 * nW, W, C
        x = self.layerNorm(x) 
        
        qkv_ea = rearrange(self.proj_ea(x + self.position_bias.repeat(1, 1, 1)),
                                '(B E nW) W (nP C) -> E nP (B nW) W C', B=B, E=3, nP=3) # nP: Q, K, V-> 3
        qkv_ea = rearrange(qkv_ea, 'E nP B W (C nH) -> E nP B nH W C', nH = self.num_heads)                     
        
        # Exp level 2 (ref)
        exposure_att1 = self.attention(qkv_ea[1,0], qkv_ea[0,1], qkv_ea[0,2], (B_//3, W_, C_), mask = attn_mask)
        exposure_att3 = self.attention(qkv_ea[1,0], qkv_ea[2,1], qkv_ea[2,2], (B_//3, W_, C_), mask = attn_mask)
        # Exp level 1
        exposure_att2to1 = self.attention(qkv_ea[0,0], qkv_ea[1,1], qkv_ea[1,2], (B_//3, W_, C_), mask = attn_mask)
        exposure_att2to1_1 = self.attention(qkv_ea[0,0], qkv_ea[2,1], qkv_ea[2,2], (B_//3, W_, C_), mask = attn_mask)
        # Exp level 3
        exposure_att2to3 = self.attention(qkv_ea[2,0], qkv_ea[1,1], qkv_ea[1,2], (B_//3, W_, C_), mask = attn_mask)
        exposure_att2to3_1 = self.attention(qkv_ea[2,0], qkv_ea[0,1], qkv_ea[0,2], (B_//3, W_, C_), mask = attn_mask)

        if self.is_shared:
            x = self.mlp1(torch.cat((exposure_att1, exposure_att3,exposure_att2to1, exposure_att2to1_1,exposure_att2to3, exposure_att2to3_1), dim=-1))
            
            x = rearrange(x, 'B W (N C)->N B W C', N=3)
        else:
            if self.keep_query:
                att2 = torch.cat((exposure_att1, exposure_att3, rearrange(qkv_ea[1,0], 'B nH W C -> B W (nH C)', nH = self.num_heads)), dim=-1)
                hidden_exp1 = self.mlp_exp1(torch.cat((exposure_att2to1, exposure_att2to1_1, rearrange(qkv_ea[0,0], 'B nH W C -> B W (nH C)', nH = self.num_heads)), dim=-1))
                hidden_exp3 = self.mlp_exp3(torch.cat((exposure_att2to3, exposure_att2to3_1, rearrange(qkv_ea[2,0], 'B nH W C -> B W (nH C)', nH = self.num_heads)), dim=-1))
            else:
                att2 = torch.cat((exposure_att1, exposure_att3), dim = -1)
                hidden_exp1 = self.mlp_exp1(torch.cat((exposure_att2to1, exposure_att2to1_1), dim = -1))
                hidden_exp3 = self.mlp_exp3(torch.cat((exposure_att2to3, exposure_att2to3_1), dim=-1))
            hidden_exp2 = self.mlp_exp2(att2) # : (B*nW, W, c)
            x = torch.cat((hidden_exp1, hidden_exp2, hidden_exp3), dim = 0)
            x = self.layerNorm(x) 

        x = x.view(3, B, Hp//self.window_size[0], Wp//self.window_size[1], 
                   self.window_size[0], self.window_size[1], self.exp_channels) # nW -> nH * nW (width)
        if self.ffn_dconv:
            x = x.permute(1, 0, 6, 2, 4, 3, 5).contiguous().view(B, 3*self.exp_channels, Hp, Wp) 
            x = self.conv11(x)
            x = rearrange(x, 'B (N C) H W -> B N C H W', N=3)
        else:
            x = x.permute(1, 0, 6, 2, 4, 3, 5).contiguous().view(B, 3, self.exp_channels, Hp, Wp) 

        if self.is_mask:
            x = torch.roll(x, shifts=(4, 4), dims=(3, 4))
        
        if pad_r > 0 or pad_b > 0:
            x = x[:,:,:,:H,:W].contiguous()

        x = x + x_origin
        shortcut = x
        x = rearrange(x, 'B N C H W -> B H W (N C)')
        x = self.layerNorm2(x)

        if self.ffn_dconv:
            x = rearrange(x, 'B H W C -> B C H W')

        x = self.mlp2(x)
        
        if self.ffn_dconv:
            x = rearrange(x, 'B (N C) H W -> B N C H W', N = 3)
        else:
            x = rearrange(x, 'B H W (N C)-> B N C H W', N = 3)

        x = shortcut + x

        if self.is_local:
            lc = self.lce(lcf)
            lc = rearrange(lc, 'B (E C) H W->B E C H W', E=3)
            x = lc + x

        return x

    def attention(self, q, k, v, x_shape, relative_position_encoding=False, mask=None):
        """ Attention operation.

        Args:
            q, k, v: query, key and value with shape of (B*nH, nW, C).
            x_shape (tuple[int]): shape of result vector.
            relative_position_encoding (bool): bias for self-attention.
        """
        B_, N, C = x_shape
        # q: B nH W C
        att_mat = (q * self.scale) @ k.transpose(-2, -1) # : (B*nW, nH, W, W)
        
        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
            att_mat = att_mat + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # B_, nH, N, N
        
        if mask is None:
            att_mat = self.softmax(att_mat)
        else:
            nW = mask.shape[0]
            att_mat = att_mat.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            att_mat = att_mat.view(-1, self.num_heads, N, N)
            att_mat = self.softmax(att_mat)
        
        return rearrange(att_mat @ v, 'B nH W C -> B W (nH C)', nH = self.num_heads)

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

class SpatialAttentionModule(nn.Module):
    def __init__(self, dim):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map
    
class Upscale(nn.Module):
    def __init__(self, n_feat):
        super(Upscale, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        x = self.body(x)
        return x

class HDRTransformer(nn.Module):
    def __init__(self, rank ,fnum=32, ref_selection=0, num_blocks=[], keep_query=False, ffn_dconv=False, is_shared=True, add_level=False, is_local=False):
        super(HDRTransformer, self).__init__()

        heads = [2,4,8]
        window_size = (8,8,8)
        num_blocks = [2,4,2]
        keep_query = keep_query
        self.rank = rank
        self.add_level = add_level
        self.ref_selection = ref_selection
        self.f1_num = fnum

        self.demosaicing = True
        self.add_level = False

        if False:
            self.conv_x1 = nn.Conv2d(4, fnum, 3, 1, 1)
            self.conv_x2 = nn.Conv2d(8, fnum, 3, 1, 1)
            self.conv_x3 = nn.Conv2d(4, fnum, 3, 1, 1)
        else:
            self.conv_x1 = nn.Conv2d(8, fnum, 3, 1, 1)
            self.conv_x2 = nn.Conv2d(16, fnum, 3, 1, 1)
            self.conv_x3 = nn.Conv2d(8, fnum, 3, 1, 1)

        self.upscale_x1 = Upscale(fnum)
        self.upscale_x2 = Upscale(fnum)
        self.upscale_x3 = Upscale(fnum)

        # Demosaicing
        if self.demosaicing:
            self.first_deconv = nn.ConvTranspose2d(self.f1_num, self.f1_num*2, kernel_size = (3,3), padding = 1)
            sr1_feats = self.f1_num*2
            sr1_kernel_size = 3
            sr1_reduction = 16
            sr1_n_resblocks = 3 
            sr1_n_resgroups = 2 
            modules_sr1 = [
                ResidualGroup(
                    default_conv, sr1_feats, sr1_kernel_size, sr1_reduction, sr1_n_resblocks) \
                for _ in range(sr1_n_resgroups)]
            self.sr1 = nn.Sequential(*modules_sr1)
            self.sr_deconv = nn.ConvTranspose2d(sr1_feats, sr1_feats, kernel_size = (3,3), stride = 2, padding = 1, output_padding=1)
            self.sr_conv = nn.Conv2d(sr1_feats, out_channels = self.f1_num, kernel_size = (3,3), padding = 'same')
        else:
            self.sr_deconv = nn.ConvTranspose2d(self.f1_num, self.f1_num, kernel_size = (3,3), stride = 2, padding = 1, output_padding=1)

        # Transformer part
        self.transformerBlock11 = nn.Sequential(*[TransformerBlock(num_heads=heads[0], window_size=window_size, fnum=self.f1_num, idx=i, keep_query=keep_query, ffn_dconv=ffn_dconv, is_shared=is_shared, is_local=is_local) 
                                            for i in range(num_blocks[0])])
        self.down1 = Downsample(self.f1_num)                                
        self.transformerBlock12 = nn.Sequential(*[TransformerBlock(num_heads=heads[1], window_size=window_size, fnum=self.f1_num*2, idx=i, keep_query=keep_query, ffn_dconv=ffn_dconv, is_shared=is_shared, is_local=is_local)  
                                            for i in range(num_blocks[1])])
        self.down2 = Downsample(self.f1_num * 2)

        if self.add_level:
            self.transformerBlock13 = nn.Sequential(*[TransformerBlock(num_heads=heads[2], window_size=window_size, fnum=self.f1_num*4, idx=i, keep_query=keep_query, ffn_dconv=ffn_dconv, is_shared=is_shared, is_local=is_local)  
                                                for i in range(num_blocks[2])])   
            self.down3 = Downsample(self.f1_num * 4)

            self.transformerBlock14 = nn.Sequential(*[TransformerBlock(num_heads=heads[3], window_size=window_size, fnum=self.f1_num*8, idx=i, keep_query=keep_query, ffn_dconv=ffn_dconv, is_shared=is_shared, is_local=is_local)  
                                                for i in range(num_blocks[3])])   
            self.up3 = Upsample(self.f1_num * 8)                                             

            self.transformerBlock23 = nn.Sequential(*[TransformerBlock(num_heads=heads[2], window_size=window_size, fnum=self.f1_num*4, idx=i, keep_query=keep_query, ffn_dconv=ffn_dconv, is_shared=is_shared, is_local=is_local) 
                                                for i in range(num_blocks[2])])       
            self.up2 = Upsample(self.f1_num * 4)
        else:
            self.transformerBlock13 = nn.Sequential(*[TransformerBlock(num_heads=heads[2], window_size=window_size, fnum=self.f1_num*4, idx=i, keep_query=keep_query, ffn_dconv=ffn_dconv, is_shared=is_shared, is_local=is_local) 
                                                for i in range(num_blocks[2])])                                    
            self.up2 = Upsample(self.f1_num * 4)

        self.transformerBlock22 = nn.Sequential(*[TransformerBlock(num_heads=heads[1], window_size=window_size, fnum=self.f1_num*2, idx=i, keep_query=keep_query, ffn_dconv=ffn_dconv, is_shared=is_shared, is_local=is_local) 
                                            for i in range(num_blocks[1])])
        self.up1 = Upsample(self.f1_num * 2)
        self.transformerBlock21 = nn.Sequential(*[TransformerBlock(num_heads=heads[0], window_size=window_size, fnum=self.f1_num, idx=i, keep_query=keep_query, ffn_dconv=ffn_dconv, is_shared=is_shared, is_local=is_local) 
                                            for i in range(num_blocks[0])])
        
        if self.add_level:
            self.reduce_chan_level3 = nn.Conv2d(self.f1_num*8, self.f1_num*4, kernel_size=1, bias=False)
        self.reduce_chan_level2 = nn.Conv2d(self.f1_num*4, self.f1_num*2, kernel_size=1, bias=False)
        self.reduce_chan_level1 = nn.Conv2d(self.f1_num*2, self.f1_num, kernel_size=1, bias=False)
        
        self.conv_last = nn.Conv2d(3*self.f1_num, self.f1_num*3, 3, 1, 1)
        self.conv_residual = nn.Conv2d(3*self.f1_num, self.f1_num*2, 3, 1, 1)

        #self.relu = nn.LeakyReLU(0.1)
        #self.norm = nn.LayerNorm(self.f1_num)

        self.upscale = nn.Sequential(#nn.Conv2d(in_channels = self.f1_num*3, out_channels = self.f1_num*2, kernel_size = (3,3), padding = 'same'),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels = self.f1_num*2, out_channels = self.f1_num*4, kernel_size = (3,3), padding = 'same'),
                                     nn.PixelShuffle(2),
                                     nn.Conv2d(in_channels = self.f1_num, out_channels = 3, kernel_size = (3,3), padding = 'same'))

    def forward(self, x):
        # b,h,w = x.shape # x: (B, H, W)
        # # Channel-wise and exposure-wise subsampling: (b,h,w) -> (b,4,4,h/4,w/4)
        # sub_x = torch.zeros(b,4,4,h//4,w//4).cuda(self.rank)#.to(torch.device('cuda'))
        # for level in range(4):
        #     offset_h = level//2
        #     offset_w = level%2
        #     sub_x[:,level,0,:,:] = x[:,offset_h::4,offset_w::4]
        #     sub_x[:,level,1,:,:] = x[:,offset_h::4,2+offset_w::4]
        #     sub_x[:,level,2,:,:] = x[:,2+offset_h::4,offset_w::4]
        #     sub_x[:,level,3,:,:] = x[:,2+offset_h::4,2+offset_w::4]

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

        x1 = sub_x[:,0,...] # x1: (B 4 H/4 W/4)
        x3 = sub_x[:,3,...]
        x2 = torch.cat((sub_x[:,1,...],sub_x[:,2,...]),dim=1) # x2: (B 8 H/4 W/4)

        f1 = self.conv_x1(x1)
        f2 = self.conv_x2(x2)
        f3 = self.conv_x3(x3)

        res_f1 = self.upscale_x1(f1)
        res_f2 = self.upscale_x2(f2)
        res_f3 = self.upscale_x3(f3)
        res_f = torch.stack((res_f1, res_f2, res_f3), dim=1)

        x = torch.stack((f1, f2, f3), dim=1)
        
        # Demosaicing
        # TODO: seperate demosaic module?
        x = rearrange(x, 'B E C H W->(B E) C H W', E = 3, C = self.f1_num)
        if self.demosaicing:
            x = self.first_deconv(x)
            sr_x = self.sr1(x)
            x = sr_x + x
            x = self.sr_deconv(x)
            x = self.sr_conv(x)
        else:
            x = self.sr_deconv(x)
        x = rearrange(x, '(B E) C H W -> B E C H W', E = 3)
        
        # HDR imaging (or put as input)
        # x[:,1,...] = x[:,1,...]/4.0
        # x[:,1,...] = x[:,1,...]/16.0
        
        # H*W scale
        x_scale11_out = self.transformerBlock11(x)

        # H//2 * W//2 scale
        x_scale12 = self.down1(x_scale11_out) 
        x_scale12_out = self.transformerBlock12(x_scale12)
        
        # H//4 * W//4 scale
        x_scale13 = self.down2(x_scale12_out) 
        x_scale13_out = self.transformerBlock13(x_scale13)
        
        if self.add_level:
            x_scale14 = self.down3(x_scale13_out)
            x_scale14 = self.transformerBlock14(x_scale14)

            x_scale14_out = self.up3(x_scale14)
            x_scale23 = self.reduce_chan_level3(rearrange(torch.cat((x_scale14_out, x_scale13_out), dim= 2), 'B E C H W -> (B E) C H W',))
            x_scale23 = rearrange(x_scale23, '(B E) C H W -> B E C H W', E = 3)
            x_scale23 = self.transformerBlock23(x_scale23)

            x_scale23_out = self.up2(x_scale23)
            x_scale22 = self.reduce_chan_level2(rearrange(torch.cat((x_scale23_out, x_scale12_out), dim= 2), 'B E C H W -> (B E) C H W',))
            x_scale22 = rearrange(x_scale22, '(B E) C H W -> B E C H W', E = 3)
            x_scale22 = self.transformerBlock22(x_scale22)

        else:
            # H//2 * W//2 scale
            x_scale13_out = self.up2(x_scale13_out)
            x_scale22 = self.reduce_chan_level2(rearrange(torch.cat((x_scale13_out, x_scale12_out), dim= 2), 'B E C H W -> (B E) C H W',))
            x_scale22 = rearrange(x_scale22, '(B E) C H W -> B E C H W', E = 3)
            x_scale22 = self.transformerBlock22(x_scale22)    
        
        # H*W scale
        x_scale22_out = self.up1(x_scale22) 
        x_scale21 = self.reduce_chan_level1(rearrange(torch.cat((x_scale22_out, x_scale11_out), dim= 2), 'B E C H W -> (B E) C H W',))
        x_scale21 = rearrange(x_scale21, '(B E) C H W -> B E C H W', E = 3)
        x_scale21 = self.transformerBlock21(x_scale21)

        # LayerNorm
        if False:
            x_scale21 = self.norm(rearrange(x_scale21, 'B E C H W -> B E H W C', E = 3))
            x_scale21 = rearrange(x_scale21, 'B E H W C -> B E C H W')

        # HDR reconstruction
        x = self.conv_last(rearrange(x + x_scale21, 'B E C H W -> B (E C) H W'))
        x = rearrange(x, 'B (E C) H W->B E C H W', E=3)
        # print(x.shape,res_f.shape)
        x = self.conv_residual(rearrange(x + res_f, 'B E C H W -> B (E C) H W'))
        x = self.upscale(x)
        x = torch.sigmoid(x)

        return x