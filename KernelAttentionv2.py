def ka_window_partition(x, window_size):
    """
    input: (B, C, H, W)
    output: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(2, 4, 0, 1, 3, 5).contiguous().view(-1, B, C, window_size, window_size)
    return windows


def ka_window_reverse(windows, window_size, H, W):
    """
    input: (num_windows*B, C, window_size, window_size)
    output: (B, C, H, W)
    """
    temp = int(windows.shape[0] / (H * W / window_size / window_size))
    if temp == 0:
        B = 1
    else:
        B = temp
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
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





# TODO: 改写代码结构，多个窗口放入写成for layer in self.conv1layers:，然后再for layer in self.selayers:，最后再cat

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

    def __init__(self, dim, num_heads, ka_window_size=16, kernel_size=3, kernel_dim_scale=1, stride=1, padding=1, K=2, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(KernelAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.img_size = 64
        self.window_size = ka_window_size
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.kernel_dim_scale = kernel_dim_scale

        self.scale = qk_scale or (dim//num_heads) ** (-0.5)

        self.hidden_dim = int(dim * kernel_dim_scale)
        self.win_num = (self.img_size//self.window_size)**2

        self.norm = nn.LayerNorm(dim)

        self.num_layers = self.win_num
        self.conv1layers = nn.ModuleList()
        self.selayers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DynamicConv(self.dim, self.kernel_size, stride=stride, padding=padding, K=K)
            self.conv1layers.append(layer)
        for i_layer in range(self.num_layers):
            layer = SELayer(self.dim)
            self.selayers.append(layer)

        self.proj_qkv = nn.Linear(self.dim, self.dim*3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_out = nn.Linear(self.dim, self.dim)


    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape

        shortcut = x

        # x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # x_windows:  win_num, bs, c, win_size, win_size
        x_windows = ka_window_partition(x, self.window_size)
        x_windows_origin = x_windows
        
        i = 0
        kernels = []
        # kernels1:  bs*out_c, k_size**2, in_c
        for layer in self.conv1layers:
            _, kernel = layer(x_windows[i].squeeze(0), kernels=None)
            kernels.append(kernel)
            i = i + 1


        # 下面想要计算所有卷积核间的自注意力
        # kernels:  bs*out_c, win_num*k_size**2, in_c
        kernels = torch.cat(kernels, 1)
        # print("k1", kernels.shape)

        # kernels_qkv:  3, bs*out_c, win_num*kernel_size**2, in_c
        kernels_qkv = self.proj_qkv(kernels).reshape(B*self.dim, self.win_num*self.kernel_size**2, 3, self.num_heads, self.dim//self.num_heads).permute(2, 0, 3, 1, 4)
        # kernels_qkv = self.proj_qkv(kernels).reshape(bs*self.dim, self.dim, 3, 4*self.kernel_size**2).permute(2, 0, 3, 1)

        # bs*out_c, win_num*kernel_size**2, in_c
        kernels_q, kernels_k, kernels_v = kernels_qkv[0], kernels_qkv[1], kernels_qkv[2]
        kernels_q = kernels_q * self.scale

        attn = (kernels_q @ kernels_k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # kernels:  bs*out_c, win_num, kernel_size**2, in_c
        kernels = (attn @ kernels_v).transpose(1, 2).reshape(B*self.dim, self.win_num*self.kernel_size**2, self.dim)

        # kernels:  win_num, bs*out_c, in_c, k_size, k_size
        kernels = self.proj_out(kernels).reshape(B*self.dim, self.win_num, self.kernel_size, self.kernel_size, self.dim).permute(1, 0, 4, 2, 3)

        # x_windows:  win_num, 1, bs*c, k_size, k_size
        x_windows_origin = x_windows_origin.view(self.win_num, 1, -1, x_windows.shape[-2], x_windows.shape[-1])

        # kernels:  bs*out_c, in_c, k_size, k_size
        i = 0
        x_windows_out = []

        for layer in self.selayers:
            # kernel:  bs*out_c, k_size**2, in_c
            kernel = layer(kernels[i])
            # x_windows:  1, bs*c, win_size, win_size
            x_window = F.conv2d(x_windows_origin[i], weight=kernel, bias=None, stride=self.stride, padding=self.padding, groups=B)
            # x_windows:  bs, c, win_size, win_size
            x_window = x_window.view(B, self.dim, self.window_size, self.window_size).unsqueeze(1)
            x_windows_out.append(x_window)
            i = i + 1

        # x_windows1:  bs*win_num, c, win_size, win_size
        x_windows = torch.cat(x_windows_out, 1).view(-1, self.dim, self.window_size, self.window_size)

        x_windows = ka_window_reverse(x_windows, self.window_size, H, W).permute(0, 3, 1, 2)

        # x_windows = shortcut + x_windows

        return x_windows
    

