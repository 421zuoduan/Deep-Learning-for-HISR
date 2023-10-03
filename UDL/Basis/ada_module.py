from UDL.Basis.self_attn_module import *
from UDL.Basis.module import *
import torch.nn.functional as F



class DWAttention(nn.Module):
    '''
    标准的SA，用以复现pytorch底层实现，即IPT中的SA
    nn.functional.multi_head_attention_forward()
    nn.MultiheadAttention

    Dynamic Convolution with Self-attention
    '''

    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        '''
        xxx -[: dim: dim*k*k, 考虑不要MLP, 用自适应卷积代替   -> MLP: 3*dim*k*k = 27*[16, 32, 64, 128, 256, 320]太大了]-
        dim: C
        num_heads: dim->[n_heads, dim//n_heads]
        qk_scale: Attention matrix 退火
        输入:  [bs, p, n_h*n_w, k*k, C]则是W-MSA, 考虑n_h*n_w可不可以放进C里， 放进k里就退化成普通SA了
        W-MSA： 进行ChannelMLP,得到 [bs,p, num_heads, n_h*n_w, k*k, k*k] （A矩阵的秩rank） TODO: k太小会影响性能？
               [bs,p, n_h*n_w, k*k, k*k] @ [bs, p, n_h*n_w, k*k, C]
        '''
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self._reset_parameters()

    def forward(self, x):
        # bs=1, p=1, n_H*n_w, k*k*C
        # B = B*p
        B, n, N, C = x.shape  # 256, 1, bs, 576

        # q, k, v = F.linear(x, self.in_proj_weight).chunk(3, dim=-1)
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q * self.scale
        # p可以变共享，放入B中
        q = q.contiguous().view(B, n, N, self.num_heads, C // self.num_heads).transpose(2, 3)
        k = k.contiguous().view(B, n, N, self.num_heads, C // self.num_heads).transpose(2, 3)
        v = v.contiguous().view(B, n, N, self.num_heads, C // self.num_heads).transpose(2, 3)
        # 1 * bs*12， 256, 48 @ 1 * bs*12, 48, 256 -> 12*bs*1, 256, 256
        attn_output_weights = q @ k.transpose(-2, -1)
        attn_output_weights = attn_output_weights.softmax(dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)
        # 12*bs, 256, 256@ bs*12, 256, 48-> bs*12, 256, 48
        #  bs*12, 256, 48-> 256, bs*12, 48 -> 256, bs, 576
        attn = (attn_output_weights @ v).transpose(2, 3).contiguous()
        attn = attn.view(B, n, N, C).contiguous()
        attn = self.out_proj(attn)
        attn = self.proj_drop(attn)

        return attn

    def _reset_parameters(self):
        # nn.init.xavier_normal_(self.q_proj.weight)
        # nn.init.xavier_normal_(self.kv_proj.weight)
        nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.q_proj.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.kv_proj.weight, a=math.sqrt(5))
        if hasattr(self.q_proj, 'bias') and self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if hasattr(self.kv_proj, 'bias') and self.kv_proj.bias is not None:
            nn.init.constant_(self.kv_proj.bias, 0.)
        if hasattr(self.out_proj, 'bias') and self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if hasattr(self, 'in_proj_bias'):
            nn.init.constant_(self.in_proj_bias, 0.)

class DWAttentionV2(nn.Module):
    '''
    标准的SA，用以复现pytorch底层实现，即IPT中的SA
    nn.functional.multi_head_attention_forward()
    nn.MultiheadAttention

    Dynamic Convolution with Self-attention
    '''

    def __init__(self, dim, patch_size, mlp_ratio, kernel_size, stride, padding,
                 num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        '''
        xxx -[: dim: dim*k*k, 考虑不要MLP, 用自适应卷积代替   -> MLP: 3*dim*k*k = 27*[16, 32, 64, 128, 256, 320]太大了]-
        dim: C
        num_heads: dim->[n_heads, dim//n_heads]
        qk_scale: Attention matrix 退火
        输入:  [bs, p, n_h*n_w, k*k, C]则是W-MSA, 考虑n_h*n_w可不可以放进C里， 放进k里就退化成普通SA了
        W-MSA： 进行ChannelMLP,得到 [bs,p, num_heads, n_h*n_w, k*k, k*k] （A矩阵的秩rank） TODO: k太小会影响性能？
               [bs,p, n_h*n_w, k*k, k*k] @ [bs, p, n_h*n_w, k*k, C]
        '''
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.patch_size = patch_size
        self.qkv_proj = LAConv(dim, dim * 3, patch_size, mlp_ratio, kernel_size, stride, padding)
        # self.kv_proj = LAConv(dim, dim * 2, patch_size, mlp_ratio, kernel_size, stride, padding)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self._reset_parameters()

    def forward(self, x):
        # bs=1, p=1, n_H*n_w, k*k*C
        # B = B*p
        H = W = self.patch_size
        B, N, C = x.shape  # 256, 1, bs, 576
        x = x.reshape(-1, H, W, C).permute(0, 3, 1, 2)
        q, k, v = self.qkv_proj(x, chunk=3)#.chunk(3, dim=1)
        q = q * self.scale
        # p可以变共享，放入B中
        q = q.contiguous().view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.contiguous().view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.contiguous().view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        # 1 * bs*12， 256, 48 @ 1 * bs*12, 48, 256 -> 12*bs*1, 256, 256
        attn_output_weights = q @ k.transpose(-2, -1)
        attn_output_weights = attn_output_weights.softmax(dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)
        # 12*bs, 256, 256@ bs*12, 256, 48-> bs*12, 256, 48
        #  bs*12, 256, 48-> 256, bs*12, 48 -> 256, bs, 576
        attn = (attn_output_weights @ v).transpose(1, 2).contiguous()
        attn = attn.view(B, N, C).contiguous()
        attn = self.out_proj(attn)
        attn = self.proj_drop(attn)

        return attn

    def _reset_parameters(self):
        # nn.init.xavier_normal_(self.q_proj.weight)
        # nn.init.xavier_normal_(self.kv_proj.weight)
        nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.q_proj.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.kv_proj.weight, a=math.sqrt(5))
        # if hasattr(self.q_proj, 'bias') and self.q_proj.bias is not None:
        #     nn.init.constant_(self.q_proj.bias, 0.)
        # if hasattr(self.kv_proj, 'bias') and self.kv_proj.bias is not None:
        #     nn.init.constant_(self.kv_proj.bias, 0.)
        if hasattr(self.out_proj, 'bias') and self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if hasattr(self, 'in_proj_bias'):
            nn.init.constant_(self.in_proj_bias, 0.)


class DPAttention(nn.Module):
    '''
    标准的SA，用以复现pytorch底层实现，即IPT中的SA
    nn.functional.multi_head_attention_forward()
    nn.MultiheadAttention

    Dynamic Convulution with Self-attention
    '''

    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        '''
        dim: dim*k*k, 考虑不要MLP, 用自适应卷积代替   -> MLP: 3*dim*k*k = 27*[16, 32, 64, 128, 256, 320]太大了
        num_heads: C->[n_heads, C//n_heads]
        qk_scale: Attention matrix 退火
        输入: [bs, p, n_h*n_w, C*k*k]  或者 [bs, p, n_h*n_w, k*k, C]则是W-MSA
        Patch-MSA： 进行ChannelMLP,得到 [bs,p, num_heads, n_h*n_w, n_h*n_w] （A矩阵意义不同, patch级别的了, 影响了rank-矩阵秩）
               考虑: [-1, num_heads, C*k*k]的A@v
               [bs,p, n_h*n_w, n_h*n_w] @ [bs, p, n_h*n_w, C*k*k]
        '''
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self._reset_parameters()

    def forward(self, x):
        # bs=1, p=1, n_H*n_w, k*k*C

        # x = x.transpose(0, 2)
        B, p, N, C = x.shape  # 256, 1, bs, 576

        # q, k, v = F.linear(x, self.in_proj_weight).chunk(3, dim=-1)
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q * self.scale
        # 如果是 N*self.num_heads, rank应该更大
        q = q.contiguous().view(B*p, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.contiguous().view(B*p, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.contiguous().view(B*p, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        # 1 * bs*12， 256, 48 @ 1 * bs*12, 48, 256 -> 12*bs*1, 256, 256
        attn_output_weights = q @ k.transpose(-1, -2)  # (q @ k.transpose(-2, -1))
        attn_output_weights = attn_output_weights.softmax(dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)
        # 12*bs, 256, 256@ bs*12, 256, 48-> bs*12, 256, 48
        #  bs*12, 256, 48-> 256, bs*12, 48 -> 256, bs, 576
        attn = (attn_output_weights @ v).transpose(1, 2).contiguous()
        attn = attn.view(B, p, N, C).contiguous()
        attn = self.out_proj(attn)
        attn = self.proj_drop(attn)
        # attn_output_weights = attn_output_weights.view(N, self.num_heads, p, B, k.size(1))# bs, 12, 256, 256
        return attn  # , None#attn_output_weights.sum(dim=1) / self.num_heads

    def _reset_parameters(self):
        # nn.init.xavier_normal_(self.q_proj.weight)
        # nn.init.xavier_normal_(self.kv_proj.weight)
        nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.q_proj.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.kv_proj.weight, a=math.sqrt(5))
        if hasattr(self.q_proj, 'bias') and self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if hasattr(self.kv_proj, 'bias') and self.kv_proj.bias is not None:
            nn.init.constant_(self.kv_proj.bias, 0.)
        if hasattr(self.out_proj, 'bias') and self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if hasattr(self, 'in_proj_bias'):
            nn.init.constant_(self.in_proj_bias, 0.)


class DPABlock(nn.Module):

    def __init__(self, dim, patch_size=3, patch_stride=3, kernel_size=3, stride=1, padding=1, num_heads=8, norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 mlp_ratio=1, sr_ratio=1, num_patch=1):
        '''
        dim: 输入通道
        window：
            输入: [bs*p, C, H, W]
            unfold: [bs*p, n_H*n_W, C*k*k]
            patch_size: 生成的window的大小, num_patches = (img_size - patch_size + 2*0) / stride + 1, 非输入的stride, 这里等于patch_size
            P-MSA: num_heads,... sr_ratio
                   输入: [bs, p=1, num_patches, k*k*C], 因此SA外部是保持k*k*C不变，内部可变
        自适应卷积: kernel_size, stride, padding 代替 MLP
                   输入通道 dim, 输出通道: dim （不同）
                   的BottleNeck，得到: [bs, p, C, H, W] -> view [bs, p, n_h*n_w, C*k*k]
        MLP: [[dim*k*k, hidden_dim], [hidden_dim, , dim*k*k]] TODO: 太大了
        fold: [bs, p, n_h*n_w, C*k*k] -> [bs, p, N, C]
        return [bs, p, N, C]
        '''

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        embed_dim = dim * patch_size * patch_size
        self.embed_dim = embed_dim
        hidden_dim = mlp_ratio * dim
        self.patch_size = patch_size

        if patch_size == patch_stride:
            print("patch-level adaptive weights")
            # BottleNeck
            self.attention1 = nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, dim, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            print("pixel-level adaptive weights")
            raise NotImplementedError()
            patch_stride = 1
            # BottleNeck
            self.attention1 = nn.Sequential(
                nn.Conv2d(dim, embed_dim, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.Sigmoid()
            )

        self.patch_stride = patch_stride
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.encoder = DPAttention(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)

        # self.linear_encoding = nn.Linear(embed_dim, embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(proj_drop),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(proj_drop)
        )
        self.norm_pre = norm_layer(embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)


        self.kernel_size = kernel_size


    def forward(self, x, H, W):
        '''
        48: O-i+2*p / s + 1 = 16 * 16, 256x9*32
        24: O-i+2*p / s + 1 = 8 * 8, 64x9*64
        12: O-i+2*p / s + 1 = 4 *4, 16x9*128
        6: O-i+2*p / s + 1 = 2 * 2, 4x9*320
        Novelty： Local Windows(like conv kernel)， 基于改进SA做动态更新
        input: bs, p, N, C 这里的p是之前层切分产生的，默认transformer相当于p=1
        -> bs*p, N, C 上层切分和bs是一个意思，已经不属于当前感受野了

        '''
        # 48x48, 24x24, 12x12, 6x6
        bs, p, N, C = x.shape  # bs, 1, 48*48, 32
        m = H // self.patch_size
        H_p = W_p = self.patch_size
        embed_dim = self.embed_dim

        # bs, p, N, C -> bs*p, C, H, W
        # 1, 1, 16*16, 64 -> 1, 64, 16, 16
        x = x.permute(0, 1, 3, 2).reshape(bs * p, C, H, W)

        # 将feature map构造成卷积核，1,64, 5*5, 3, 3, (16 - 3 + 0)/3 + 1 =  5 new_patch
        # 1, 64*25, 9 (bs, C*k*k, n_H*n_W) -> (bs, n_H*n_W, C*k*k)
        y = F.unfold(x, H_p, stride=self.patch_stride).permute(0, 2, 1)  # .permute(0, 2, 3, 1).contiguous()  # .transpose(0, 1)
        # NC一起做的LayerNorm， 接着对NC做MLP投影（一般SA则是Channel MLP）， 如果不是3x3, 也可以做Conv，Encoder里很少有人用Conv代替MLP
        # 1, 1, 25, 9C 不同patch要有不同的MLP的weights
        # x = x + self.linear_encoding(self.norm_pre(x))

        # Adaptive Convolution
        x = self.attention1(x)  # bs*p, k*k, H, W  -> bs*p, C, n_H*n_W, k*k 无需让通道=k**2
        x = x.reshape(bs * p, H // H_p, H_p, W // W_p, W_p, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(bs * p, -1,
                                                                                                           H_p * W_p * C)
        x = x * y
        # Attention要兼容 bs=1, p=1, n_H*n_w, k*k*C
        x = x.reshape(bs, p, -1, self.embed_dim)
        x = x + self.encoder(self.norm1(x))[0]

        x = self.mlp_head(self.norm2(x)) + x
        # 256, p, bs, 9*32
        x = x.view(bs * p, -1, self.embed_dim)
        # b*p, n_h*n+w, k*k*C
        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), H, (H_p, W_p),
                                     stride=(H_p, W_p))

        x = x.reshape(bs, p, C, N).permute(0, 1, 3, 2)

        return x


class DWABlock(nn.Module):

    def __init__(self, dim, patch_size=3, kernel_size=3, stride=1, padding=1, num_heads=8, norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 mlp_ratio=1, sr_ratio=1, num_patch=1):
        '''
        dim: 输入通道
        window：
            输入: [bs*p, C, H, W]
            unfold: [bs*p, n_H*n_W, C*k*k]
            patch_size: 生成的window的大小, num_patches = (img_size - patch_size + 2*0) / stride + 1, 非输入的stride, 这里等于patch_size
            W-MSA: num_heads,... sr_ratio
                   输入: [bs, p=1, num_patches, k*k, C]
                   输出: [bs, p=1, num_patches, k*k, C]
                   受限制于残差连接: SA输入输出有相同的通道数，但是内部可以变化

        自适应卷积: kernel_size, stride, padding 代替 MLP
                   输入通道 dim, 输出通道: dim （不同）
                   的BottleNeck，得到: [bs, p, C, H, W] -> view [bs, p, n_h*n_w, C*k*k] * unfold 的 x
                   DDF的不适用，DDF没有考虑有unfold(x)的情况
        MLP: [[dim, hidden_dim], [hidden_dim, , dim]]
        fold: [bs, p, n_h*n_w, C*k*k] -> [bs, p, N, C]
        return [bs, p, N, C]
        '''

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        # embed_dim = dim * patch_size * patch_size
        # self.embed_dim = embed_dim
        hidden_dim = mlp_ratio * dim
        self.patch_size = patch_size

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.encoder = DWAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)

        # self.linear_encoding = nn.Linear(embed_dim, embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Dropout(proj_drop),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop)
        )
        self.norm_pre = norm_layer(hidden_dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # BottleNeck
        self.attention1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kernel_size = kernel_size

    def forward(self, x, H, W):
        '''
        48: O-i+2*p / s + 1 = 16 * 16, 256x9*32
        24: O-i+2*p / s + 1 = 8 * 8, 64x9*64
        12: O-i+2*p / s + 1 = 4 *4, 16x9*128
        6: O-i+2*p / s + 1 = 2 * 2, 4x9*320
        Novelty： Local Windows(like conv kernel)， 基于改进SA做动态更新
        input: bs, p, N, C 这里的p是之前层切分产生的，默认transformer相当于p=1
        -> bs*p, N, C 上层切分和bs是一个意思，已经不属于当前感受野了

        '''
        # 48x48, 24x24, 12x12, 6x6
        bs, p, N, C = x.shape  # bs, 1, 48*48, 32
        m = H // self.patch_size
        H_p = W_p = self.patch_size
        dim = self.dim

        # bs, p, N, C -> bs*p, C, H, W
        # 1, 1, 16*16, 64 -> 1, 64, 16, 16
        x = x.permute(0, 1, 3, 2).reshape(bs * p, C, H, W)

        # 将feature map构造成卷积核，1,64, 5*5, 3, 3, (16 - 3 + 0)/3 + 1 =  5 new_patch
        # 1, 64*25, 9 (bs, C*k*k, n_H*n_W) -> (bs, n_H*n_W, C*k*k)
        y = F.unfold(x, H_p, stride=H_p).permute(0, 2, 1)

        # Adaptive Convolution
        x = self.attention1(x)  # bs*p, k*k, H, W  -> bs*p, C, n_H*n_W, k*k 无需让通道=k**2
        x = x.reshape(bs * p, H // H_p, H_p, W // W_p, W_p, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(bs * p, -1,
                                                                                                           H_p * W_p * C)
        x = x * y
        # Attention要兼容 bs=1, p=1, n_H*n_w, k*k*C
        x = x.reshape(bs * p, -1, H_p*W_p, C)
        x = x + self.encoder(self.norm1(x))[0]

        x = self.mlp_head(self.norm2(x)) + x
        # 256, p, bs, 9*32
        x = x.view(bs * p, -1, H_p*W_p*C)
        # b*p, n_h*n+w, k*k*C
        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), H, (H_p, W_p),
                                     stride=(H_p, W_p))

        x = x.reshape(bs, p, C, N).permute(0, 1, 3, 2)

        return x

class DWABlockV2(nn.Module):

    def __init__(self, dim, patch_size=3, kernel_size=3, stride=1, padding=1, num_heads=8, norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 mlp_ratio=1, sr_ratio=1, num_patch=1):
        '''
        dim: 输入通道
        window：
            输入: [bs*p, C, H, W]
            unfold: [bs*p, n_H*n_W, C*k*k]
            patch_size: 生成的window的大小, num_patches = (img_size - patch_size + 2*0) / stride + 1, 非输入的stride, 这里等于patch_size
            W-MSA: num_heads,... sr_ratio
                   输入: [bs, p=1, num_patches, k*k, C]
                   输出: [bs, p=1, num_patches, k*k, C]
                   受限制于残差连接: SA输入输出有相同的通道数，但是内部可以变化

        自适应卷积: kernel_size, stride, padding 代替 MLP
                   输入通道 dim, 输出通道: dim （不同）
                   的BottleNeck，得到: [bs, p, C, H, W] -> view [bs, p, n_h*n_w, C*k*k] * unfold 的 x
                   DDF的不适用，DDF没有考虑有unfold(x)的情况
        MLP: [[dim, hidden_dim], [hidden_dim, , dim]]
        fold: [bs, p, n_h*n_w, C*k*k] -> [bs, p, N, C]
        return [bs, p, N, C]
        '''

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        # embed_dim = dim * patch_size * patch_size
        # self.embed_dim = embed_dim
        hidden_dim = mlp_ratio * dim
        self.patch_size = patch_size

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.encoder = DWAttentionV2(dim, patch_size, mlp_ratio, kernel_size, stride, padding,
                                     num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)

        # self.linear_encoding = nn.Linear(embed_dim, embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Dropout(proj_drop),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop)
        )
        self.norm_pre = norm_layer(hidden_dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # BottleNeck
        self.laconv = LAConv(dim=dim, outdim=dim, patch_size=patch_size, mlp_ratio=mlp_ratio, kernel_size=kernel_size, padding=padding, stride=stride)
        self.kernel_size = kernel_size

    def forward(self, x, H, W):
        '''
        48: O-i+2*p / s + 1 = 16 * 16, 256x9*32
        24: O-i+2*p / s + 1 = 8 * 8, 64x9*64
        12: O-i+2*p / s + 1 = 4 *4, 16x9*128
        6: O-i+2*p / s + 1 = 2 * 2, 4x9*320
        Novelty： Local Windows(like conv kernel)， 基于改进SA做动态更新
        input: bs, p, N, C 这里的p是之前层切分产生的，默认transformer相当于p=1
        -> bs*p, N, C 上层切分和bs是一个意思，已经不属于当前感受野了

        '''
        # 48x48, 24x24, 12x12, 6x6
        bs, p, N, C = x.shape  # bs, 1, 48*48, 32
        m = H // self.patch_size
        H_p = W_p = self.patch_size
        dim = self.dim

        # bs, p, N, C -> bs*p, C, H, W
        # 1, 1, 16*16, 64 -> 1, 64, 16, 16
        x = x.permute(0, 1, 3, 2).reshape(bs * p, C, H, W)

        # 将feature map构造成卷积核，1,64, 5*5, 3, 3, (16 - 3 + 0)/3 + 1 =  5 new_patch
        # Adaptive Convolution
        x = self.laconv(x) # bs*p, C*n_W*n_W, k*k*C

        # Attention要兼容 bs=1, p=1, n_H*n_w, k*k*C
        x = x.reshape(-1, H_p*W_p, C)

        x = x + self.encoder(x)[0]

        x = self.mlp_head(self.norm2(x)) + x
        # 256, p, bs, 9*32
        x = x.view(bs * p, -1, H_p*W_p*C)
        # b*p, n_h*n+w, k*k*C
        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), H, (H_p, W_p),
                                     stride=(H_p, W_p))

        x = x.reshape(bs, p, C, N).permute(0, 1, 3, 2)

        return x


class LAConv(nn.Module):
    def __init__(self, dim, outdim, patch_size, mlp_ratio=4, kernel_size=3, stride=1, padding=1, bias=True, proj_mode='conv'):
        super().__init__()
        self.patch_size = patch_size
        hidden_dim = dim * mlp_ratio
        self.D = outdim
        if proj_mode == 'dwconv':
            self.attention1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim, bias=bias),
                nn.GELU()
            )
        else:
            self.attention1 = nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, outdim, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x, chunk=1, order='N'):
        patch = self.patch_size
        B, C, H, W = x.shape
        y = F.unfold(x, patch, stride=patch).permute(0, 2, 1)
        x = self.attention1(x)
        x = x.reshape(B, self.D, H // patch, patch, W // patch, patch).permute(0, 2, 4, 3, 5, 1).contiguous().view(B, -1, patch * patch * self.D)
        if chunk == 1:
            if order == 'N':
                return x * y
            elif order == 'C':
                return (x * y).view(B, patch, patch, self.D).permute(0, 3, 1, 2)
        else:
            qkv = x.chunk(3, dim=-1)
            return [(x*y).reshape(-1, patch, patch, C).permute(0, 3, 1, 2).contiguous() for x in qkv]



def compared_shapred_weight():
    # 通道维度是共享weights的， 不是卷积那样
    x = torch.ones([1, 2, 3, 5])
    # y = x@W.T+b, [3,5] x [5, 2]
    mlp = nn.Linear(in_features=5, out_features=2)
    y = mlp(x)

    print(mlp.weight.t())

    print(y)
    print('-' * 100)

    #
    x = torch.ones([1, 2, 5, 5])
    # Kernel Weights: [2,2,k,k] 同一个out_channels里不同in_channels是不一样的，不同的也是不一样的
    # w0 * [5, 5] + w1 * [5, 5] 有2个
    conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
    y = conv(x)

    print(conv.weight, conv.weight.shape)

    print(y)


def test_module():

    x = torch.randn([8, 1, 32*32, 16]).cuda()
    module = DWABlockV2(dim=16, patch_size=16, kernel_size=3, stride=1, padding=1, num_heads=4,
             qkv_bias=True, mlp_ratio=3).cuda()
    y = module(x, 32, 32)

    print(y.shape)


if __name__ == '__main__':
    test_module()





