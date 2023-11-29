import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum


class FastLeFF(nn.Module):
    
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop = 0.):
        super().__init__()

        from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs × hw × c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        x = self.dwconv(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)
        x = self.linear2(x)

        return x
    

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

        
######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False, bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v    


class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q, k, v


class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

            
        if token_projection =='conv':
            self.qkv = ConvProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        elif token_projection =='linear':
            self.qkv = LinearProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


########### self-attention #############
class Attention(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
            
        self.qkv = LinearProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs × hw × c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        x = self.dwconv(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)
        x = self.linear2(x)
        x = self.eca(x)

        return x


def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1), stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B', C, Wh, Ww
        windows = windows.permute(0,2,3,1).contiguous() # B', Wh, Ww, C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B', Wh, Ww, C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B', Wh, Ww, C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1), stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out
    

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=34, out_channel=48, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x



def ka_window_partition(x, window_size):
    """
    input: (B, H*W, C)
    output: (B, num_windows*C, window_size, window_size)
    """
    B, L, C = x.shape
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(B, -1, window_size, window_size)
    return windows


def ka_window_reverse(windows, window_size, H, W):
    """
    input: (B, num_windows*C, window_size, window_size)
    output: (B, H*W, C)
    """
    B = windows.shape[0]
    x = windows.contiguous().view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, H*W, -1)
    return x


class SELayer_KA(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(channel, channel // 16 if channel >= 64 else channel, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(channel // 16 if channel >= 64 else channel, channel, kernel_size=1),
                                nn.Sigmoid(), )

    def forward(self, x):
        channel_weight = self.se(x)
        x = x * channel_weight
        return x
    

class WinKernel_Reweight(nn.Module):
    def __init__(self, dim, win_num=4):
        super().__init__()
        
        self.dim = dim
        self.win_num = win_num
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.downchannel = nn.Conv2d(win_num*dim, win_num, kernel_size=1, groups=win_num)
        self.linear1 = nn.Conv2d(win_num, win_num*4, kernel_size=1)
        self.gelu = nn.GELU()
        self.linear2 = nn.Conv2d(win_num*4, win_num, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, kernels, windows):
        """
        kernels:  win_num*c, c, k ,k
        windows:  bs, win_num*c, wh, ww
        """

        B = windows.shape[0]

        # win_weight:  bs, win_num*c, 1, 1
        win_weight = self.pooling(windows)

        # win_weight:  bs, win_num, 1, 1
        win_weight = self.downchannel(win_weight)

        win_weight = self.linear1(win_weight)
        win_weight = win_weight.permute(0, 2, 3, 1).reshape(B, 1, -1)
        win_weight = self.gelu(win_weight)
        win_weight = win_weight.transpose(1, 2).reshape(B, -1, 1, 1)
        win_weight = self.linear2(win_weight)
        # weight:  bs, win_num, 1, 1, 1, 1
        weight = self.sigmoid(win_weight).unsqueeze(-1).unsqueeze(-1)

        # kernels:  1, win_num, c, c, k, k
        kernels = kernels.reshape(self.win_num, self.dim, self.dim, kernels.shape[-2], kernels.shape[-1]).unsqueeze(0)

        # kernels:  bs, win_num, c, c, k, k
        kernels = kernels.repeat(B, 1, 1, 1, 1, 1)

        # kernels:  bs, win_num, c, c, k, k
        kernels = weight * kernels

        # kernels:  bs*win_num*c, c, k, k
        kernels = kernels.reshape(-1, self.dim, kernels.shape[-2], kernels.shape[-1])

        return kernels
    

class ConvLayer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, groups=4, win_num=4, k_in=False):
        super().__init__()

        self.dim = dim
        self.win_num = win_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        if not k_in:
            self.params = nn.Parameter(torch.randn(win_num*dim, dim, kernel_size, kernel_size), requires_grad=True)
        else:
            self.params = None

    def forward(self, x, kernels=None, groups=None):
        '''
        x:  bs, win_num*c, wh, ww
        kernels:  None

        or

        x:  bs, win_num*c, h, w
        kernels:  c*win_num, c, k_size, k_size
        '''

        if kernels is None:
            x = F.conv2d(x, self.params, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            x = F.conv2d(x, kernels, stride=self.stride, padding=self.padding, groups=groups)

        return x, self.params


class KernelAttention(nn.Module):
    """
    第一个分组卷积产生核，然后计算核的自注意力，调整核，第二个分组卷积产生输出，skip connection
    
    Args:
        dim: 输入通道数
        window_size: 窗口大小
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        qk_scale: 缩放因子
        attn_drop: 注意力dropout
        proj_drop: 输出dropout
        ka_window_size: kernel attention window size
        kernel_size: 卷积核大小
        stride: 卷积步长
        padding: 卷积padding
    """

    def __init__(self, dim, input_resolution, num_heads, ka_win_num=4, kernel_size=3, stride=1, padding=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_num = ka_win_num
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.scale = qk_scale or (dim//num_heads) ** (-0.5)
        self.window_size = int(input_resolution // math.sqrt(ka_win_num))

        self.num_layers = self.win_num
        self.convlayer1 = ConvLayer(dim, kernel_size, stride, padding, groups=ka_win_num, k_in=False)
        self.wink_reweight = WinKernel_Reweight(dim, win_num=ka_win_num)
        self.gk_generation = nn.Conv2d(self.win_num*self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.convlayer2 = ConvLayer(dim, kernel_size, stride, padding, k_in=True)
        self.fusion = nn.Conv2d((self.win_num+1)*self.dim, self.dim, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        """
        x: B, L, C
        """
        B, L, C = x.shape
        H, W = self.input_resolution, self.input_resolution

        # x_windows:  bs, win_num*c, wh, ww
        x_windows = ka_window_partition(x, self.window_size)

        # windows_conv1:  bs, win_num*c, wh, ww
        # kernels:  win_num*c, c, k_size, k_size
        windows_conv1, kernels = self.convlayer1(x_windows)


        ### 给窗口卷积核赋权        A1win1 ... A4win4
        # kernels:  bs*win_num*c, c, k_size, k_size
        kernels = self.wink_reweight(kernels, windows_conv1)


        ### 生成全局卷积核global kernel
        # kernels:  bs*c, win_num*c, k_size, k_size
        kernels = kernels.reshape(B, self.win_num, self.dim, self.dim, self.kernel_size, self.kernel_size).transpose(1, 2).reshape(B*self.dim, self.win_num*self.dim, self.kernel_size, self.kernel_size)

        # global_kernel:  bs*c, c, k_size, k_size
        global_kernel = self.gk_generation(kernels)

        # global_kernel:  bs, 1, c, c, k_size, k_size
        global_kernel = global_kernel.reshape(B, self.dim, self.dim, self.kernel_size, self.kernel_size).unsqueeze(1)

        # kernels:  bs, win_num, c, c, k_size, k_size
        kernels = kernels.reshape(B, self.dim, self.win_num, self.dim, self.kernel_size, self.kernel_size).transpose(1, 2)

        # kernels:  bs, win_num+1, c, c, k_size, k_size
        kernels = torch.cat([kernels, global_kernel], dim=1)

        # kernels:  bs*(win_num+1)*c, c, k_size, k_size
        kernels = kernels.reshape(-1, self.dim, self.kernel_size, self.kernel_size)


        ### 卷积核与输入特征计算卷积
        # x:  bs, c, h, w
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # x:  bs, (win_num+1)*c, h, w
        x = x.repeat(1, self.win_num+1, 1, 1)

        # x:  1, bs*(win_num+1)*c, h, w
        x = x.reshape(1, B*(self.win_num+1)*C, H, W)

        # x:  1, bs*(win_num+1)*c, h, w
        x, _ = self.convlayer2(x, kernels, groups=B*(self.win_num+1))
        
        # x:  bs, (win_num+1)*c, h, w
        x = x.reshape(B, (self.win_num+1)*C, H, W)

        # x:  bs, c, h, w
        x = self.fusion(x)

        # x:  bs, h*w, c
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)

        return x


#########################################
########### Transformer #############
class TransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, ka_win_num=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear', token_mlp='leff'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp

        self.norm1 = norm_layer(dim)

        branch_num_heads = num_heads if (num_heads//2)!=0 else 1

        self.kernelattention = KernelAttention(dim//2, input_resolution[0], num_heads=branch_num_heads, ka_win_num=ka_win_num, kernel_size=3, stride=1, padding=1, qk_scale=qk_scale)

        self.attn = WindowAttention(
            dim//2, win_size=to_2tuple(self.win_size), num_heads=branch_num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn','mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) 
        elif token_mlp=='leff':
            self.mlp =  LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        elif token_mlp=='fastleff':
            self.mlp =  FastLeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)    
        else:
            raise Exception("FFN error!") 

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        shortcut = x
        # x: B, L, C
        x = self.norm1(x)

        x_wa, x_ka = x.chunk(2, dim=-1)
        
        # Kernel Attention
        x_ka = x_ka.view(B, L, C//2)
        x_ka = self.kernelattention(x_ka)

        # Window Attention
        x_wa = x_wa.view(B, H, W, C//2)

        # partition windows
        x_windows = window_partition(x_wa, self.win_size)  # nW*B, win_size, win_size, C/2  N*C/2->C/2
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C//2)  # nW*B, win_size*win_size, C/2

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, win_size*win_size, C/2

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C//2)
        x_wa = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C/2
        x_wa = x_wa.view(B, H * W, C//2)

        # concat wa and ka
        x = torch.cat([x_wa, x_ka], dim=-1)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#########################################
########### Basic layer ################
class BasicLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 token_projection='linear', token_mlp='ffn'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, win_size=win_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Block(nn.Module):
    def __init__(self, img_size=256, in_chans=34,
                 embed_dim=48, depths=[2, 2, 2, 2, 2], num_heads=[1, 2, 4, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='mlp',
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[2]
        dec_dpr = enc_dpr[::-1]

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=embed_dim, kernel_size=3, stride=1)
        
        # Encoder
        self.encoderlayer_0 = BasicLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            token_projection=token_projection, token_mlp=token_mlp)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            token_projection=token_projection, token_mlp=token_mlp)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)

        # Bottleneck
        self.conv = BasicLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            token_projection=token_projection, token_mlp=token_mlp)

        # Decoder
        self.upsample_1 = upsample(embed_dim*4, embed_dim*2)
        self.decoderlayer_1 = BasicLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[0:depths[3]],
                            norm_layer=norm_layer,
                            token_projection=token_projection, token_mlp=token_mlp)
        self.upsample_0 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_0 = BasicLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[3:3]):sum(depths[3:4])],
                            norm_layer=norm_layer,
                            token_projection=token_projection, token_mlp=token_mlp)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, H, W, x):

        # Input Projection
        y = self.input_proj(x)
        # x = x.permute(0,2,3,1).reshape(x.shape[0], -1, x.shape[1])
        y = self.pos_drop(y)
        #Encoder
        conv0 = self.encoderlayer_0(y)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.dowsample_1(conv1)

        # Bottleneck
        conv = self.conv(pool1)

        #Decoder
        up1 = self.upsample_1(conv)
        deconv1 = torch.cat([up1, conv1], -1)
        deconv1 = self.decoderlayer_1(deconv1)

        up0 = self.upsample_0(deconv1)
        deconv0 = torch.cat([up0, conv0], -1)
        deconv0 = self.decoderlayer_0(deconv0)

        # Output Projection
        y = self.output_proj(deconv0)
        return y


if __name__ == "__main__":
    input_size = 64
    arch = Block
    depths=[2, 2, 2, 2, 2]
    model_restoration = Block(img_size=input_size, embed_dim=32, depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='mlp')
    print(model_restoration)
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('# model_restoration parameters: %.2f M'%(sum(param.numel() for param in model_restoration.parameters())/ 1e6))
    print("number of GFLOPs: %.2f G"%(model_restoration.flops() / 1e9))