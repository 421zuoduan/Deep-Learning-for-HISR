import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from math import sqrt

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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Attention(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

def ka_window_partition(x, window_size):
    """
    input: (B, H*W, C)
    output: (B, num_windows*C, window_size, window_size)
    """
    B, L, C = x.shape
    H, W = int(sqrt(L)), int(sqrt(L))
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
        kernels:  win_num*c, 1, k ,k
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

        # kernels:  1, win_num, c, 1, k, k
        kernels = kernels.reshape(self.win_num, self.dim, 1, kernels.shape[-2], kernels.shape[-1]).unsqueeze(0)

        # kernels:  bs, win_num, c, 1, k, k
        kernels = kernels.repeat(B, 1, 1, 1, 1, 1)

        # kernels:  bs, win_num, c, 1, k, k
        kernels = weight * kernels

        # kernels:  bs*win_num*c, 1, k, k
        kernels = kernels.reshape(-1, 1, kernels.shape[-2], kernels.shape[-1])

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
            self.params = nn.Parameter(torch.randn(win_num*dim, 1, kernel_size, kernel_size), requires_grad=True)
        else:
            self.params = None

    def forward(self, x, kernels=None, groups=None):
        '''
        x:  bs, (win_num+1)*c, wh, ww
        kernels:  None

        or

        x:  bs, (win_num+1)*c, h, w
        kernels:  c*win_num, c, k_size, k_size
        '''

        if kernels is None:
            C = x.shape[1] // self.win_num
            x = F.conv2d(x, self.params, stride=self.stride, padding=self.padding, groups=self.groups*C)
        else:
            C = x.shape[1] // (self.win_num+1)
            # x:  1, bs*(win_num+1)*c, h, w
            x = x.reshape(1, -1, x.shape[-2], x.shape[-1])
            x = F.conv2d(x, kernels, stride=self.stride, padding=self.padding, groups=groups*C)

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

    def __init__(self, dim, input_resolution, num_heads, ka_win_num=4, kernel_size=7, stride=1, padding=3, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_num = ka_win_num
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.scale = qk_scale or (dim//num_heads) ** (-0.5)
        self.window_size = int(input_resolution // sqrt(ka_win_num))

        self.num_layers = self.win_num
        self.convlayer1 = ConvLayer(dim, kernel_size, stride, padding, groups=ka_win_num, k_in=False)
        self.wink_reweight = WinKernel_Reweight(dim, win_num=ka_win_num)
        self.gk_generation = nn.Conv2d(self.win_num, 1, kernel_size=1, stride=1, padding=0)
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
        # kernels:  win_num*c, 1, k_size, k_size
        windows_conv1, kernels = self.convlayer1(x_windows)


        ### 给窗口卷积核赋权        A1win1 ... A4win4
        # kernels:  bs*win_num*c, 1, k_size, k_size
        kernels = self.wink_reweight(kernels, windows_conv1)


        ### 生成全局卷积核global kernel
        # kernels:  bs*c, win_num, k_size, k_size
        kernels = kernels.reshape(B, self.win_num, self.dim, 1, self.kernel_size, self.kernel_size).transpose(1, 2).reshape(B*self.dim, self.win_num, self.kernel_size, self.kernel_size)

        # global_kernel:  bs*c, 1, k_size, k_size
        global_kernel = self.gk_generation(kernels)

        # global_kernel:  bs, 1, c, 1, k_size, k_size
        global_kernel = global_kernel.reshape(B, self.dim, 1, self.kernel_size, self.kernel_size).unsqueeze(1)

        # kernels:  bs, win_num, c, 1, k_size, k_size
        kernels = kernels.reshape(B, self.dim, self.win_num, 1, self.kernel_size, self.kernel_size).transpose(1, 2)

        # kernels:  bs, win_num+1, c, 1, k_size, k_size
        kernels = torch.cat([kernels, global_kernel], dim=1)

        # kernels:  bs*(win_num+1)*c, 1, k_size, k_size
        kernels = kernels.reshape(-1, 1, self.kernel_size, self.kernel_size)


        ### 卷积核与输入特征计算卷积
        # x:  bs, c, h, w
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # x:  bs, (win_num+1)*c, h, w
        x = x.repeat(1, self.win_num+1, 1, 1)

        # x:  1, bs*(win_num+1)*c, h, w
        x, _ = self.convlayer2(x, kernels, groups=B*(self.win_num+1))
        
        # x:  bs, (win_num+1)*c, h, w
        x = x.reshape(B, (self.win_num+1)*C, H, W)

        # x:  bs, c, h, w
        x = self.fusion(x)

        # x:  bs, h*w, c
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)

        return x



class Window_Attention(nn.Module):
    r""" PSRT Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim=32, input_resolution=16, num_heads=8, window_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=4, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ka_win_num=4):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.window_size <= input_resolution, "input_resolution should be larger than window_size"

        self.KernelAttention = KernelAttention(dim//2, input_resolution, num_heads=num_heads//2, ka_win_num=ka_win_num, kernel_size=7, stride=1, padding=3)
        self.attn = Attention(
            dim//2, num_heads=num_heads//2,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self,H, W, x):
        x = rearrange(x, 'B C H W -> B (H W) C')
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = x.view(B, H, W, C)
        x = self.norm1(x)

        x_wa, x_ka = x.chunk(2, dim=-1)
        x_ka = x_ka.view(B, H*W, C//2)
        x_ka = self.KernelAttention(x_ka)

        # partition windows
        x_windows = window_partition(x_wa, self.window_size)  # nW*B, window_size, window_size, C/2

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C//2)  # nW*B, window_size*window_size, C/2

        # attention
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C/2

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C//2)
        x_wa = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C/2
        x_wa = x_wa.view(B, H * W, C//2)
        x = torch.cat([x_wa, x_ka], dim=-1)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))


        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x


class Window_Attention_Shuffle(nn.Module):
    r""" PSRT Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim=32, input_resolution=16, num_heads=8, window_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=4, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.window_size <= input_resolution, "input_resolution should be larger than window_size"

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, H, W, x):
        x = rearrange(x, 'B C H W -> B (H W) C')
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = x.view(B, H, W, C)
        x = self.norm1(x)
        # shuffle
        x = rearrange(x, 'B H W C -> B C H W')
        x = Win_Shuffle(x, self.window_size)
        x = rearrange(x, 'B C H W -> B H W C')

        # partition windows

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # attention
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x


class Window_Attention_Reshuffle(nn.Module):
    r""" PSRT Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim=32, input_resolution=16, num_heads=8, window_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=4, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.window_size <= input_resolution, "input_resolution should be larger than window_size"

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, H, W, x):
        x = rearrange(x, 'B C H W -> B (H W) C')
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = x.view(B, H, W, C)
        x = self.norm1(x)
        # shuffle
        x = rearrange(x, 'B H W C -> B C H W')
        x = Win_Reshuffle(x, self.window_size)
        x = rearrange(x, 'B C H W -> B H W C')

        # partition windows

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # attention
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x

def Win_Shuffle(x, win_size):
    """
    :param x: B C H W
    :param win_size:
    :return: y: B C H W
    """
    B, C, H, W = x.shape
    dilation = win_size // 2
    resolution = H
    assert resolution % win_size == 0, 'resolution of input should be divisible'
    assert win_size % 2 == 0, 'win_size should be the multiple of two'
    "input size BxCxHxW"
    "shuffle"

    N1 = H // dilation
    N2 = W // dilation
    x = rearrange(x, 'B C H W -> B H W C')
    x = window_partition(x, dilation)  # BN x d x d x c
    x = x.reshape(-1, N1, N2, C * dilation ** 2)  # B x n x n x d2c
    xt = torch.zeros_like(x)
    x0 = x[:, 0::2, 0::2, :]  # B n/2 n/2 d2c
    x1 = x[:, 0::2, 1::2, :]  # B n/2 n/2 d2c
    x2 = x[:, 1::2, 0::2, :]  # B n/2 n/2 d2c
    x3 = x[:, 1::2, 1::2, :]  # B n/2 n/2 d2c

    xt[:, 0:N1 // 2, 0:N2 // 2, :] = x0  # B n/2 n/2 d2c
    xt[:, 0:N1 // 2, N2 // 2:N2, :] = x1  # B n/2 n/2 d2c
    xt[:, N1 // 2:N1, 0:N2 // 2, :] = x2  # B n/2 n/2 d2c
    xt[:, N1 // 2:N1, N2 // 2:N2, :] = x3  # B n/2 n/2 d2c
    xt = xt.reshape(-1, dilation, dilation, C)
    xt = window_reverse(xt, dilation, H, W)
    xt = rearrange(xt, 'B H W C -> B C H W')

    return xt

def Win_Reshuffle(x, win_size):
    """
        :param x: B C H W
        :param win_size:
        :return: y: B C H W
        """
    B, C, H, W = x.shape
    dilation = win_size // 2
    N1 = H // dilation
    N2 = W // dilation
    assert H % win_size == 0, 'resolution of input should be divisible'
    assert win_size % 2 == 0, 'win_size should be the multiple of two'

    x = rearrange(x, 'B C H W -> B H W C')
    x = window_partition(x, dilation)  # BN x d x d x c
    x = x.reshape(-1, N1, N2, C * dilation ** 2)  # B x n x n x d2c
    xt = torch.zeros_like(x)
    xt[:, 0::2, 0::2, :] = x[:, 0:N1// 2, 0:N2 // 2, :]  # B n/2 n/2 d2c
    xt[:, 0::2, 1::2, :] = x[:, 0:N1 // 2, N2 // 2:N2, :]  # B n/2 n/2 d2c
    xt[:, 1::2, 0::2, :] = x[:, N1 // 2:N1, 0:N2 // 2, :]  # B n/2 n/2 d2c
    xt[:, 1::2, 1::2, :] = x[:, N1 // 2:N1, N2 // 2:N2, :]  # B n/2 n/2 d2c
    xt = xt.reshape(-1, dilation, dilation, C)
    xt = window_reverse(xt, dilation, H, W)
    xt = rearrange(xt, 'B H W C -> B C H W')

    return xt

class SaR_Block(nn.Module):
    def __init__(self, img_size=64, in_chans=32, head=8, win_size=4, norm_layer=nn.LayerNorm):
        """
        input: B x F x H x W
        :param img_size: size of image
        :param in_chans: feature of image
        :param embed_dim:
        :param token_dim:
        """
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_chans
        self.win_size = win_size
        self.norm2 = norm_layer(in_chans)
        self.norm3 = norm_layer(in_chans)
        self.WA1 = Window_Attention(dim=self.in_channels, input_resolution=img_size, num_heads=head,
                                    window_size=win_size)
        self.WA2 = Window_Attention(dim=self.in_channels, input_resolution=img_size, num_heads=head,
                                    window_size=win_size)
        self.WA3 = Window_Attention(dim=self.in_channels, input_resolution=img_size, num_heads=head,
                                    window_size=win_size)
        # self.WA2 = Window_Attention_Shuffle(dim=self.in_channels, input_resolution=img_size, num_heads=head,
        #                             window_size=win_size)
        # self.WA3 = Window_Attention_Reshuffle(dim=self.in_channels, input_resolution=img_size, num_heads=head,
        #                             window_size=win_size)


    def forward(self, H, W, x):
        # window_attention1
        shortcut = x
        x = self.WA1(H, W, x)

        # shuffle
        # window_attention2
        x = self.WA2(H, W, x)

        # reshuffle
        # window_attention3
        x = self.WA3(H, W, x)

        x = x + shortcut

        return x

class PSRT_Block(nn.Module):
    def __init__(self, num=3, img_size=64, in_chans=32, head=8, win_size=8):
        """
        input: B x H x W x F
        :param img_size: size of image
        :param in_chans: feature of image
        :param num: num of layer
        """
        super().__init__()
        self.num_layers = num
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SaR_Block(img_size=img_size, in_chans=in_chans, head=head, win_size=win_size//(2**i_layer))
            self.layers.append(layer)

    def forward(self, H, W, x):
        for layer in self.layers:
            x = layer(H, W, x)
        return x

class Block(nn.Module):
    def __init__(self, out_num, inside_num, img_size, in_chans, embed_dim, head, win_size):
        super().__init__()
        self.num_layers = out_num
        self.layers = nn.ModuleList()
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        for i_layer in range(self.num_layers):
            layer = PSRT_Block(num=inside_num, img_size=img_size, in_chans=embed_dim, head=head, win_size=win_size)
            self.layers.append(layer)

    def forward(self,H, W, x):
        x = self.conv(x)
        for layer in self.layers:
            x = layer(H, W, x)
        return x

if __name__ == '__main__':
    import time
    start = time.time()
    input = torch.randn(1, 32, 64, 64)
    encoder = Block(out_num=2, inside_num=3, img_size=64, in_chans=32, embed_dim=32, head=8, win_size=8)
    output = encoder(64, 64, input)
    print(output.shape)
