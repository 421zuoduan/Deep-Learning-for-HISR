import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

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

def ka_window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size,
               C)
    windows = x.permute(1, 3, 0, 2, 4,
                        5).contiguous().view(-1, B, C, window_size, window_size)
    return windows


def ka_window_reverse(windows, window_size, H, W):
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
    temp = int(windows.shape[0] / (H * W / window_size / window_size))
    if temp == 0:
        B = 1
    else:
        B = temp
    x = windows.contiguous().view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, H, W, -1)
    return x

class KernelWeight(nn.Module):
    def __init__(self, dim, K):
        super(KernelWeight, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, K, 1)
        self.fc2 = nn.Conv2d(K, K, 1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x, 1)
    

class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(channel, channel // 16 if channel >= 64 else channel, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(channel // 16 if channel >= 64 else channel, channel, kernel_size=1),
                                nn.Sigmoid(), )

    def forward(self, x):
        channel_weight = self.se(x)
        x = x * channel_weight
        return x


class DynamicConv(nn.Module):
    def __init__(self, dim, kernel_size=5, stride=2, padding=1, dilation=1, groups=1, bias=True, K=4):
        super(DynamicConv, self).__init__()

        assert dim%groups==0
        self.in_planes = dim
        self.out_planes = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = KernelWeight(dim, K)

        self.weight = nn.Parameter(torch.Tensor(K, dim, dim//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, dim))
        else:
            self.bias = None


    def forward(self, x, kernels=None):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2)*output.size(-1))
        return output, aggregate_weight.view(batch_size*self.out_planes, self.in_planes, -1).permute(0, 2, 1)


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
        bs: batch size
        ka_window_size: kernel attention window size
        kernel_size: 卷积核大小
        kernel_dim_scale: 卷积核通道数缩放因子
        stride: 卷积步长
        padding: 卷积padding
    """

    def __init__(self, dim, num_heads, ka_window_size=32, kernel_size=5, kernel_dim_scale=1, stride=1, padding=2, K=2, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(KernelAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.img_size = 64
        self.window_size = self.img_size // 2
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.kernel_dim_scale = kernel_dim_scale

        self.scale = qk_scale or (dim//num_heads) ** (-0.5)

        self.hidden_dim = int(dim * kernel_dim_scale)
        self.window_num = (self.img_size//self.window_size)**2

        self.dynamic_conv1 = DynamicConv(self.dim, self.kernel_size, stride=stride, padding=padding, K=K)
        self.dynamic_conv2 = DynamicConv(self.dim, self.kernel_size, stride=stride, padding=padding, K=K)
        self.dynamic_conv3 = DynamicConv(self.dim, self.kernel_size, stride=stride, padding=padding, K=K)
        self.dynamic_conv4 = DynamicConv(self.dim, self.kernel_size, stride=stride, padding=padding, K=K)

        self.se1 = SELayer(self.dim)
        self.se2 = SELayer(self.dim)
        self.se3 = SELayer(self.dim)
        self.se4 = SELayer(self.dim)

        self.proj_qkv = nn.Linear(self.dim, self.dim*3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_out = nn.Linear(self.dim, self.dim)


    def forward(self, x_windows):
        """
        x_windows: B, C, H, W
        """
        B, C, H, W = x_windows.shape

        bs = B

        # x_windows:  bs*win_num, c, win_size, win_size
        x_windows = ka_window_partition(x_windows, self.window_size)

        x_windows_chunk = torch.chunk(x_windows, 4, dim=0)
        
        # kernels1:  bs*out_c, k_size**2, in_c
        x_windows1, kernels1 = self.dynamic_conv1(x_windows_chunk[0].squeeze(0), kernels=None)
        x_windows2, kernels2 = self.dynamic_conv2(x_windows_chunk[1].squeeze(0), kernels=None)
        x_windows3, kernels3 = self.dynamic_conv3(x_windows_chunk[2].squeeze(0), kernels=None)
        x_windows4, kernels4 = self.dynamic_conv4(x_windows_chunk[3].squeeze(0), kernels=None)


        # 下面想要计算所有卷积核间的自注意力
        # kernels:  bs*out_c, 4*k_size**2, in_c
        kernels = torch.cat((kernels1.unsqueeze(1), kernels2.unsqueeze(1), kernels3.unsqueeze(1), kernels4.unsqueeze(1)), 2)

        # kernels_qkv:  3, bs*out_c, 4*kernel_size**2, in_c
        kernels_qkv = self.proj_qkv(kernels).reshape(bs*self.dim, 4*self.kernel_size**2, 3, self.num_heads, self.dim//self.num_heads).permute(2, 0, 3, 1, 4)
        # kernels_qkv = self.proj_qkv(kernels).reshape(bs*self.dim, self.dim, 3, 4*self.kernel_size**2).permute(2, 0, 3, 1)

        # bs*out_c, 4*kernel_size**2, in_c
        kernels_q, kernels_k, kernels_v = kernels_qkv[0], kernels_qkv[1], kernels_qkv[2]
        kernels_q = kernels_q * self.scale

        attn = (kernels_q @ kernels_k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # kernels:  bs*out_c, 4, kernel_size**2, in_c
        kernels = (attn @ kernels_v).transpose(1, 2).reshape(bs*self.dim, 4*self.kernel_size**2, self.dim)

        # kernels:  4, bs*out_c, in_c, k_size, k_size
        kernels = self.proj_out(kernels).reshape(bs*self.dim, 4, self.kernel_size, self.kernel_size, self.dim).permute(1, 0, 4, 2, 3)

        # kernels:  bs*out_c, in_c, k_size, k_size
        kernels1 = self.se1(kernels[0])#.view(bs*self.dim, self.dim, self.kernel_size, self.kernel_size)
        kernels2 = self.se2(kernels[1])#.view(bs*self.dim, self.dim, self.kernel_size, self.kernel_size)
        kernels3 = self.se3(kernels[2])#.view(bs*self.dim, self.dim, self.kernel_size, self.kernel_size)
        kernels4 = self.se4(kernels[3])#.view(bs*self.dim, self.dim, self.kernel_size, self.kernel_size)
        

        # x_windows1:  1, bs*c, win_size, win_size
        x_windows1 = F.conv2d(x_windows[0].view(1, -1, x_windows.shape[-2], x_windows.shape[-1]), weight=kernels1, bias=None, stride=self.stride, padding=self.padding, groups=bs)
        x_windows2 = F.conv2d(x_windows[1].view(1, -1, x_windows.shape[-2], x_windows.shape[-1]), weight=kernels2, bias=None, stride=self.stride, padding=self.padding, groups=bs)
        x_windows3 = F.conv2d(x_windows[2].view(1, -1, x_windows.shape[-2], x_windows.shape[-1]), weight=kernels3, bias=None, stride=self.stride, padding=self.padding, groups=bs)
        x_windows4 = F.conv2d(x_windows[3].view(1, -1, x_windows.shape[-2], x_windows.shape[-1]), weight=kernels4, bias=None, stride=self.stride, padding=self.padding, groups=bs)

        # # x_windows1:  bs, c, win_size, win_size
        x_windows1 = x_windows1.view(bs, self.dim, self.window_size, self.window_size)
        x_windows2 = x_windows2.view(bs, self.dim, self.window_size, self.window_size)
        x_windows3 = x_windows3.view(bs, self.dim, self.window_size, self.window_size)
        x_windows4 = x_windows4.view(bs, self.dim, self.window_size, self.window_size)

        # x_windows1:  bs*win_num, win_size, win_size, c
        x_windows = torch.cat((x_windows1.unsqueeze(1), x_windows2.unsqueeze(1), x_windows3.unsqueeze(1), x_windows4.unsqueeze(1)), 1).view(-1, self.dim, self.window_size, self.window_size)

        x_windows = ka_window_reverse(x_windows, self.window_size, H, W)

        return x_windows.permute(0, 3, 1, 2)
    


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

    def forward(self,H, W, x):
        x = rearrange(x, 'B C H W -> B (H W) C')
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = x.view(B, H, W, C)
        x = self.norm1(x)
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
        self.WA2 = Window_Attention_Shuffle(dim=self.in_channels, input_resolution=img_size, num_heads=head,
                                    window_size=win_size)
        self.WA3 = Window_Attention_Reshuffle(dim=self.in_channels, input_resolution=img_size, num_heads=head,
                                    window_size=win_size)
        self.KA = KernelAttention(dim=self.in_channels, num_heads=head, kernel_size=5, K=2)


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

        # kernel_attention
        x = self.KA(x)

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
