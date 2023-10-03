import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionFromPvt(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # 步进8卷积
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PvtBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # the effect of sr_ratio
        self.attn = AttentionFromPvt(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)  # self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.mlp(self.norm2(x), H, W)  # self.drop_path(self.mlp(self.norm2(x)))

        return x

class HAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                                     nn.Linear(dim, dim * 2, bias=qkv_bias),
                                                     nn.Dropout(attn_drop)) for i in range(num_patch)])
        # for i in range(downscaling_factor):
        #     self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #     self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #     H_module.append(self.q)
        #     H_

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.apply(self._reset_parameters)
        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        out = torch.zeros_like(x)
        # x = x.transpose(1, 2)
        B, p, N, C = x.shape

        for i, (blk, sub_x) in enumerate(zip(self.H_module, x.chunk(p, dim=1))):
            q = blk[0]
            kv = blk[1]
            attn_drop = blk[2]

            q = q(sub_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # if self.sr_ratio > 1:
            #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            #     # 步进8卷积
            #     x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            #     x_ = self.norm(x_)
            #     kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # else:
            kv = kv(sub_x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            sub_x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            sub_x = self.proj(sub_x)
            sub_x = self.proj_drop(sub_x)

            out[:, i, ...] = sub_x

        return out

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            nn.init.trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

class PAttention_Depre(nn.Module):
    '''
    QKV的W仅操作了通道方向，这适用于CP->D的情况，否则效果不好
        x_s     x_h
    QxK 1, NC x NC, p, = 1, p
    KxV 1,p x p, N, C = 1, N, C = x_s.shape
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                                     nn.Linear(dim, dim * 2, bias=qkv_bias),
                                                     nn.Dropout(attn_drop)) for i in range(num_patch)])
        # for i in range(downscaling_factor):
        #     self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #     self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #     H_module.append(self.q)
        #     H_

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    def forward(self, x_s, x_h, H, W):
        B, p, N, C = x_h.shape
        _, N_q, p_s, D = x_s.shape

        out = torch.zeros(B, N_q, 1, C).to(x_h.device)

        for i, blk in enumerate(self.H_module):
            q = blk[0]
            kv = blk[1]
            attn_drop = blk[2]

            q = q(x_s).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # q = q.reshape(B, self.num_heads, p_s, -1)  # B, heads, N_q*D

            kv = kv(x_h).reshape(B, p, N, 2, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                          5)  # 2, B, heads, p, N_q, C'= D
            kv = kv.reshape(2, B, self.num_heads, p, -1)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale  # B, heads, 1, p
            # attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            x_s = (attn @ v).transpose(1, 2).reshape(B, N_q, D)

            x_s = self.proj(x_s)
            x_s = self.proj_drop(x_s)

            out[:, :, i, ...] = x_s

        return out

class PAttention_modified(nn.Module):
    '''
    QKV的W仅操作了通道方向，这适用于CP->D的情况，否则效果不好
        x_s     x_h
    QxK 1, NC x NC, p, = 1, p
    KxV 1,p x p, N, C = 1, N, C = x_s.shape patch_aggregation
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                                     nn.Linear(dim, dim * 2, bias=qkv_bias),
                                                     nn.Dropout(attn_drop)) for i in range(num_patch)])
        # for i in range(downscaling_factor):
        #     self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #     self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #     H_module.append(self.q)
        #     H_

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._reset_parameters)
        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    def forward(self, x_s, x_h, H, W):
        B, p, N, C = x_h.shape
        _, p_s, N_q, D = x_s.shape
        # x_s = x_s.reshape(B, 1, -1)
        # N_xs = N_q * D

        # out = torch.zeros(B, p_s, N_q, C).to(x_h.device)

        for i, blk in enumerate(self.H_module):
            q = blk[0]
            kv = blk[1]
            attn_drop = blk[2]

            q = q(x_s).reshape(B * p_s, N_q, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
            q = q.reshape(B * p_s, self.num_heads, 1, -1)  # B, heads, N_q*D

            kv = kv(x_h).reshape(B, p, N, 2, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                          5)  # 2, B, heads, p, N_q, C'= D

            # q = q(x_s.reshape(B, -1)).reshape(B, self.num_heads, -1)
            # kv = kv(x_h.reshape(B, p, -1)).reshape(B, p, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 2, B, heads, p, N_q, C'= D

            kv = kv.reshape(2, B, self.num_heads, p, -1)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale  # B, heads, 1, p
            # attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            #B, h , 1, p  v:B,h,p,N*C
            # attn = attn.transpose(-1, -2)
            # x_s = (attn * v).transpose(1, 2).reshape(B, p, -1)#N_q*D
            # x_s = self.proj(x_s).reshape(B, p, N, C)

            x_s = (attn @ v).transpose(1, 2).reshape(B, p_s, N_q, D)
            x_s = self.proj(x_s)
            x_s = self.proj_drop(x_s)

            # out[:, i, ...] = x_s

        return x_s

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            nn.init.trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

class PAttention_modified_ep(nn.Module):
    '''
    QKV的W仅操作了通道方向，这适用于CP->D的情况，否则效果不好
        x_s     x_h
    QxK 1, NC x NC, p, = 1, p
    KxV 1,p x p, N, C = 1, N, C = x_s.shape patch_aggregation
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                                     nn.Linear(dim, dim * 2, bias=qkv_bias),
                                                     nn.Dropout(attn_drop)) for i in range(num_patch)])
        # for i in range(downscaling_factor):
        #     self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #     self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #     H_module.append(self.q)
        #     H_

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._reset_parameters)
        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    def forward(self, x_s, x_h, H, W):
        B, p, N, C = x_h.shape
        _, p_s, N_q, D = x_s.shape
        # x_s = x_s.reshape(B, 1, -1)
        # N_xs = N_q * D

        # out = torch.zeros(B, 1, N_q, C).to(x_h.device)

        for i, blk in enumerate(self.H_module):
            q = blk[0]
            kv = blk[1]
            attn_drop = blk[2]

            # q = q(x_s).reshape(B, N_q, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
            # q = q.reshape(B, self.num_heads, 1, -1)  # B, heads, N_q*D

            # kv = kv(x_h).reshape(B, p, N, 2, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
            #                                                                               5)  # 2, B, heads, p, N_q, C'= D

            q = q(x_s.reshape(B, -1)).reshape(B, self.num_heads, -1)
            kv = kv(x_h.reshape(B, p, -1)).reshape(B, p, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 2, B, heads, p, N_q, C'= D

            kv = kv.reshape(2, B, self.num_heads, p, -1)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale  # B, heads, 1, p
            # attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            #B, h , 1, p  v:B,h,p,N*C
            attn = attn.transpose(-1, -2)
            x_s = (attn * v).transpose(1, 2).reshape(B, p, -1)#N_q*D
            x_s = self.proj(x_s).reshape(B, p, N, C)

            # x_s = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
            # x_s = self.proj(x_s)
            # x_s = self.proj_drop(x_s)

            #out[:, i, ...] = x_s

        return x_s#out

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            nn.init.trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

class SPAttention(nn.Module):
    def __init__(self, patch_num, ch, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, 2 * dim, bias=qkv_bias),
                                                     # nn.Linear(patch_pixel, patch_pixel, bias=qkv_bias),#也可以dim, dim
                                                     nn.Dropout(attn_drop)) for i in range(patch_num)])
        # for i in range(downscaling_factor):
        #     self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #     self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #     H_module.append(self.q)
        #     H_

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(ch, ch)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._reset_parameters)
        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    def forward(self, x, x_o, H, W):
        _, _, N_q, _ = x.shape
        B, p, N, C = x_o.shape
        x_o = x_o.reshape(B, p, -1)#B, p, -1
        x = x.reshape(B, p, -1) #B,P,N*C->B,P,P, 更耗费显存，考虑对N*C做linear
        out = torch.zeros(B, p, N, C).to(x.device)
        D = N * C
        zero_mask = torch.zeros(p,p).to(x.device)
        one_mask = torch.ones(p, p).to(x.device)
        for i, blk in enumerate(self.H_module):
            q = blk[0]
            # v = blk[1]
            attn_drop = blk[1]

            # v = v(x_o).reshape(B, p, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            x_o = x_o.reshape(B, p, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

            qk = q(x).reshape(B, p, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)#2, B, heads, p, D
            q, k = qk[0], qk[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale #B, heads, p, p

            #得到dim=-1大于0.5的attn点
            # indices = (attn - 0.5 > 1e-6).nonzero()
            # attn = attn[indices]
            attn_mask = torch.where(attn > 0.5, one_mask, zero_mask)
            attn_max = torch.max(attn, dim=-1, keepdim=True).values == attn#torch.argmax(attn, dim=-1, keepdim=True)
            # attn_max = gather_nd(one_mask, attn_max_ind, [2, 3])
            attn_mask += attn_max
            attn_mask = torch.clip(attn_mask, max=1)

            # dist_mat_sorted, sorted_indices = torch.sort(attn, dim=-1)
            # top_k = torch.max(sorted_indices, dim=1)
            # knn_indexes = sorted_indices[:, :, 1:top_k]  # HWC, K
            # mask[:, X_row, attn_topk[:, base]] = 1
            patch_pos = attn * attn_mask
            pos_attn = torch.exp(patch_pos)
            attn = torch.sum(torch.exp(attn), dim=-1, keepdim=True)
            attn = pos_attn / attn
            # attn = attn.softmax(dim=-1)#knn
            attn = attn_drop(attn)
            x_h = (attn @ x_o).reshape(B, p, N, C)

            x_h = self.proj(x_h)
            x_h = self.proj_drop(x_h)

            out = x_h

        return out

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            nn.init.trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

class HBlock(nn.Module):

    def __init__(self, down_scale, norm_dim, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True,
                 hybrid_ffn=True):  # compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)  # dim
        # the effect of sr_ratio
        self.attn = HAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if hybrid_ffn:
            self.ffn = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             compatiable=compatiable)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.apply(self._init_weights)

    def forward(self, x, H, W):
        # x: (b,P,N,c)
        x = self.norm1(x)
        x = x + self.attn(x, H, W)  # self.drop_path(self.attn(self.norm1(x), H, W))
        x = self.norm2(x)
        x = x + self.ffn(x, H, W)  # self.drop_path(self.mlp(self.norm2(x)))

        return x

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

class TwoHPBlock(nn.Module):

    def __init__(self, down_scale, norm_dim, dim, num_heads, #num_patch
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True,
                 hybrid_ffn=True):  # compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)  # dim
        self.norm_h1 = norm_layer(dim if compatiable else norm_dim)  # dim
        # PAttention_modified << From idea of Huang, 小patch在大patch中查询
        self.attn = PAttention_modified(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.self_attn = HAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        self.norm3 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if hybrid_ffn:
            self.ffn = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             compatiable=compatiable)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.apply(self._init_weights)

    def forward(self, x_s, x_h, H, W):
        # x: (b,P,N,c)
        # x_s = self.norm1(x_s)
        # p_x = x_s + self.attn(x_s, self.norm_h1(x_h), H, W)
        # x = p_x + self.self_attn(self.norm3(p_x), H, W)
        # x = x + self.ffn(self.norm2(x), H, W)
        p_x = self.norm1(x_s)
        x = p_x + self.attn(p_x, self.norm_h1(x_h), H, W) + self.self_attn(p_x, H, W) # self.drop_path(self.attn(self.norm1(x), H, W))
        x = self.norm2(x)
        x = x + self.ffn(x, H, W)  # self.drop_path(self.mlp(self.norm2(x)))

        return x

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

class TwoThreeHBlock(nn.Module):

    def __init__(self, down_scale, norm_dim, dim, num_heads,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True,
                 hybrid_ffn=True):  # compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)  # dim
        self.norm_h1 = norm_layer(dim if compatiable else norm_dim)  # dim

        self.norm_embed = norm_layer(dim if compatiable else norm_dim)  # dim
        self.norm_embed_h = norm_layer(dim if compatiable else norm_dim)  # dim

        # the effect of sr_ratio
        self.attn = PAttention_modified(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.self_attn = HAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.self_attn_h = HAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        self.norm_h2 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if hybrid_ffn:
            self.ffn = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             compatiable=compatiable)
            self.ffn_h = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             compatiable=compatiable)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.ffn_h = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.apply(self._init_weights)

    def forward(self, x_s, x_h, H, W):
        '''
        self-attn x@x^T
        attn: x_s@x_h@x_h的过程
        x_s只有1个聚合的patch
        '''
        # x: (b,P,N,c)
        x_s = self.norm1(x_s)
        p_x = x_s + self.attn(x_s, self.norm_h1(x_h), H, W)
        p_x = self.norm_embed(p_x)
        x = p_x + self.self_attn(p_x, H, W)
        x_h = self.norm_embed_h(x_h)
        x_h = x_h + self.self_attn_h(x_h, H, W)
        x = self.norm2(x)
        x_h = self.norm_h2(x_h)
        x = x + self.ffn(x, H, W)
        x_h = x_h + self.ffn_h(x_h, H, W)
        return x, x_h

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

class TNTBlock(nn.Module):

    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, hybrid_ffn=True):
        super().__init__()
        self.has_inner = inner_dim > 0

        # Inner
        self.inner_norm1 = norm_layer(inner_dim)
        self.inner_attn = Attention(
            inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


        # Outer
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(
            outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)

        self.inner_norm2 = norm_layer(inner_dim)
        self.proj_norm1 = norm_layer(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
        self.proj_norm2 = norm_layer(outer_dim)
        if hybrid_ffn:
            self.inner_mlp = HyFFN(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)
        else:
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def forward(self, inner_tokens, outer_tokens, H, W):

        inner_tokens = inner_tokens + \
                       self.inner_attn(self.inner_norm1(inner_tokens))  # B*N, k*k, c
        inner_tokens = inner_tokens + \
                       self.inner_mlp(self.inner_norm2(inner_tokens), H, W)  # B*N, k*k, c,c->D
        B, N, C = outer_tokens.size()
        outer_tokens = outer_tokens + self.proj_norm2(
            self.proj(self.proj_norm1(inner_tokens.reshape(B, N, -1))))  # B, N, 8*8*C, 8*8*C->

        outer_tokens = outer_tokens + self.outer_attn(self.outer_norm1(outer_tokens))
        outer_tokens = outer_tokens + self.outer_mlp(self.outer_norm2(outer_tokens), H, W)

        return inner_tokens, outer_tokens

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class TwoThreeTNTBlock(nn.Module):

    def __init__(self, inner_dim, outer_dim, outer_num_heads, inner_num_heads, num_patches, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, hybrid_ffn=True):
        super().__init__()
        self.has_inner = inner_dim > 0
        self.num_words = num_words
        self.inner_dim = inner_dim

        self.num_patches = num_patches

        #cross
        self.attn = PAttention_modified_ep(dim=outer_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(inner_dim)
        self.norm2 = norm_layer(outer_dim)
        self.lproj_norm1 = norm_layer(num_patches * inner_dim)
        self.lproj_norm2 = norm_layer(inner_dim)
        self.lproj = nn.Linear(num_patches * inner_dim, inner_dim, bias=False)

        # Inner
        self.inner_norm1 = norm_layer(inner_dim)
        self.inner_attn = Attention(
            inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


        # Outer
        self.outer_norm1 = norm_layer(num_words * inner_dim)
        self.outer_attn = Attention(
            num_words * inner_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(num_words * inner_dim)

        self.inner_norm2 = norm_layer(inner_dim)
        self.proj_norm1 = norm_layer(outer_dim)
        self.proj = nn.Linear(outer_dim, outer_dim, bias=False)
        self.proj_norm2 = norm_layer(num_words * inner_dim)
        if hybrid_ffn:
            self.inner_mlp = HyFFN(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)
            self.mlp = HyFFN(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                   out_features=inner_dim, act_layer=act_layer, drop=drop)
        else:
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def forward(self, inner_tokens, outer_tokens, H, W):
        bs, N, H, W, C = inner_tokens.shape


        inner_tokens = inner_tokens.flatten(0, 1).flatten(1, 2)
        inner_tokens = inner_tokens + \
                       self.inner_attn(self.inner_norm1(inner_tokens))  # B*N, k*k, c -> D
        inner_tokens = inner_tokens + \
                       self.inner_mlp(self.inner_norm2(inner_tokens), H, W)  # B, k*k, c -> D
        _, _, D = inner_tokens.size()
        B, N, C = outer_tokens.size()
        outer_tokens = outer_tokens + self.proj_norm2(
            self.proj(self.proj_norm1(inner_tokens.reshape(B, N, -1))))  # B, N, 8*8*C, 8*8*C-> IPT

        outer_tokens = outer_tokens + self.outer_attn(self.outer_norm1(outer_tokens))
        outer_tokens = outer_tokens + self.outer_mlp(self.outer_norm2(outer_tokens), H, W)

        ############################
        # cross
        tokens = inner_tokens.reshape(bs, -1, self.num_words, D).permute(0, 2, 1, 3).flatten(2)
        tokens = self.lproj_norm2(
            self.lproj(self.lproj_norm1(tokens)))  # B, N, 8*8*C, 8*8*C-> IPT
        # B, 1, k*k*C @ B,P,k*k*C = B, 1, P
        # B, 1, P * B, P, k*k*C = B, 1, k*k*C
        tokens = tokens.reshape(bs, 1, self.num_words, self.inner_dim)
        outer_tokens = self.norm2(outer_tokens).reshape(bs, self.num_patches,  self.num_words, self.inner_dim)#B,P,N,C
        print("1", tokens.shape, outer_tokens.shape)
        outer_tokens = outer_tokens + self.attn(self.norm1(tokens),
                                                outer_tokens, H, W)

        return inner_tokens.reshape(bs, N, H, W, D), outer_tokens.reshape(B, -1, C)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class BSAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
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
        # 256, 1, 576
        # x = x.unsqueeze(1)
        B, p, N, C = x.shape # 256, 1, bs, 576 注意这里N是bs，且q@k->

        # q, k, v = F.linear(x, self.in_proj_weight).chunk(3, dim=-1)
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q * self.scale
        q = q.contiguous().view(B, p * N * self.num_heads, C // self.num_heads).transpose(0, 1)
        k = k.contiguous().view(B, p * N * self.num_heads, C // self.num_heads).transpose(0, 1)
        v = v.contiguous().view(B, p * N * self.num_heads, C // self.num_heads).transpose(0, 1)
        # 256， 1 * bs*12，48 @ 256， 48, 1 * bs*12 -> 12*bs*1, 1 * bs*12, 1 * bs*12
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))#(q @ k.transpose(-2, -1))
        attn_output_weights = attn_output_weights.softmax(dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)
        # 12*bs, 256, 256@ bs*12, 256, 48-> bs*12, 256, 48
        #  bs*12, 256, 48-> 256, bs*12, 48 -> 256, bs, 576
        attn = torch.bmm(attn_output_weights, v).transpose(0, 1).contiguous()
        attn = attn.view(B, p*N, C)
        attn = self.out_proj(attn)
        attn = self.proj_drop(attn)
        # attn_output_weights = attn_output_weights.view(N, self.num_heads, p, B, k.size(1))# bs, 12, 1 * bs*12, 1 * bs*12
        return attn.view(B, p, N, C), None#attn_output_weights.sum(dim=1) / self.num_heads

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
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)

class BSABlock(nn.Module):

    def __init__(self, dim, patch_dim=3, num_heads=8, norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mlp_ratio=1, sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        embed_dim = dim * patch_dim * patch_dim
        self.embed_dim = embed_dim
        hidden_dim = mlp_ratio * dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.encoder = BSAttention(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)

        self.linear_encoding = nn.Linear(embed_dim, embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(proj_drop),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(proj_drop)
        )

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

    def forward(self, x, H, W):
        '''
        48: O-i+2*p / s + 1 = 16 * 16, 256x9*32
        24: O-i+2*p / s + 1 = 8 * 8, 64x9*64
        12: O-i+2*p / s + 1 = 4 *4, 16x9*128
        6: O-i+2*p / s + 1 = 2 * 2, 4x9*320
        两个卖点： Local Windows， widely-search
        '''
        # 48x48, 24x24, 12x12, 6x6
        B, p, N, C = x.shape #bs, 1, 48*48, 32
        x = x.permute(0, 1, 3, 2).reshape(B*p, C, H, W)

        x = torch.nn.functional.unfold(x,
            self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0, 1).contiguous()
        x = x.reshape(-1, p, B, self.embed_dim)
        #256, 1, 2, 288 （N，p, bs, 9C）这里和PSA不一样，是IPT实现方式
        x = x + self.linear_encoding(x)
        src2 = self.norm1(x)
        x = x + self.encoder(src2)[0]
        x = self.norm2(x)

        x = self.mlp_head(x) + x
        # 256, p, bs, 9*32
        x = x.transpose(0, 2).contiguous().view(B*p, -1, self.embed_dim)
        # B,C,N->B,C,H,W 2，288，256:2,32*9,16,16
        x = x.transpose(1, 2).contiguous()
        print(x.shape)
        x = torch.nn.functional.fold(x, H, self.patch_dim,
                                     stride=self.patch_dim)

        x = x.reshape(B, p, C, N).permute(0, 1, 3, 2)

        return x

class PSAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
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
        # 256, 1, 576
        # x = x.unsqueeze(1)
        x = x.transpose(0, 2)
        B, p, N, C = x.shape # bs, 1, 256, 576 注意这里N是bs

        # q, k, v = F.linear(x, self.in_proj_weight).chunk(3, dim=-1)
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q * self.scale
        #p可以变共享，放入B中
        q = q.contiguous().view(B, p * N * self.num_heads, C // self.num_heads).transpose(0, 1)
        k = k.contiguous().view(B, p * N * self.num_heads, C // self.num_heads).transpose(0, 1)
        v = v.contiguous().view(B, p * N * self.num_heads, C // self.num_heads).transpose(0, 1)
        # 1 * bs*12， 256, 48 @ 1 * bs*12, 48, 256 -> 12*bs*1, 256, 256
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))#(q @ k.transpose(-2, -1))
        attn_output_weights = attn_output_weights.softmax(dim=-1)
        attn_output_weights = self.attn_drop(attn_output_weights)
        # 12*bs, 256, 256@ bs*12, 256, 48-> bs*12, 256, 48
        #  bs*12, 256, 48-> 256, bs*12, 48 -> 256, bs, 576
        attn = torch.bmm(attn_output_weights, v).transpose(0, 1).contiguous()
        attn = attn.view(B, p, N, C).transpose(0, 2).contiguous()
        attn = self.out_proj(attn)
        attn = self.proj_drop(attn)
        # attn_output_weights = attn_output_weights.view(N, self.num_heads, p, B, k.size(1))# bs, 12, 256, 256
        return attn, None#attn_output_weights.sum(dim=1) / self.num_heads

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
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)

class PSABlock(nn.Module):

    def __init__(self, dim, patch_dim=3, num_heads=8, norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mlp_ratio=1, sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        embed_dim = dim * patch_dim * patch_dim
        self.embed_dim = embed_dim
        hidden_dim = mlp_ratio * dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.encoder = PSAttention(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)

        self.linear_encoding = nn.Linear(embed_dim, embed_dim)
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

    def forward(self, x, H, W):
        '''
        48: O-i+2*p / s + 1 = 16 * 16, 256x9*32
        24: O-i+2*p / s + 1 = 8 * 8, 64x9*64
        12: O-i+2*p / s + 1 = 4 *4, 16x9*128
        6: O-i+2*p / s + 1 = 2 * 2, 4x9*320
        两个卖点： Local Windows， widely-search
        '''
        # 48x48, 24x24, 12x12, 6x6
        bs, p, N, C = x.shape #bs, 1, 48*48, 32
        if N!=1:
            x = x.permute(0, 1, 3, 2).reshape(bs*p, C, H, W)
            #B,9C,N=16*16 -> transpose B,N,9C
            # 1,1,24*24,64->1,36,1024:B,N,9C
            x = torch.nn.functional.unfold(x,
                self.patch_dim, stride=self.patch_dim).transpose(1, 2).contiguous()#.transpose(0, 1)
            # BSA: 256, 1, bs, 9C, 而PSA： bs, 1, 256, 9C
            x = x.reshape(bs, p, -1, self.embed_dim)#B,N,9C
            # TODO: 输入norm
            x = self.norm_pre(x)
            x = x + self.linear_encoding(x) # 不受影响，还是共享的，TODO：adaptive
            x = x + self.encoder(self.norm1(x))[0]
            # TODO: 输入norm
            x = self.norm2(x)
            x = self.mlp_head(x) + x
            # 256, p, bs, 9*32
            x = x.transpose(0, 2).contiguous().view(bs*p, -1, self.embed_dim)

            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), H, self.patch_dim,
                                         stride=self.patch_dim)

            x = x.reshape(bs, p, C, N).permute(0, 1, 3, 2)

            return x
        else:
            return

class HybridHPBlock(nn.Module):

    def __init__(self, down_scale, norm_dim, dim, num_heads, #num_patch
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True,
                 hybrid_ffn=True):  # compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)  # dim
        self.norm_h1 = norm_layer(dim if compatiable else norm_dim)  # dim

        self.attn = PSABlock(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.self_attn = HAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        self.norm3 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if hybrid_ffn:
            self.ffn = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             compatiable=compatiable)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.apply(self._init_weights)

    def forward(self, x_s, H, W):
        # x: (b,P,N,c)
        # x_s = self.norm1(x_s)
        # p_x = x_s + self.attn(x_s, self.norm_h1(x_h), H, W)
        # x = p_x + self.self_attn(self.norm3(p_x), H, W)
        # x = x + self.ffn(self.norm2(x), H, W)
        p_x = self.norm1(x_s)
        x = p_x + self.attn(p_x, H, W) + self.self_attn(p_x, H, W) # self.drop_path(self.attn(self.norm1(x), H, W))
        x = self.norm2(x)
        x = x + self.ffn(x, H, W)  # self.drop_path(self.mlp(self.norm2(x)))

        return x

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        if len(x.size()) == 3:
            B, N, C = x.size()
            # L = int(math.sqrt(N))
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            x = self.dwconv(x)
            x = x.reshape(B, C, N).permute(0, 2, 1).contiguous()
        else:
            b, p, N, C = x.shape
            # L = int(math.sqrt(N))
            x = x.reshape(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
            x = self.dwconv(x)
            x = x.reshape(b, p, C, -1).permute(0, 1, 3, 2).contiguous()
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
        self.hidden_features = hidden_features

        self.apply(self._reset_weights)

    def _reset_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            nn.init.trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class HyFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., compatiable=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.hidden_features = hidden_features
        self.conv = DWConv(hidden_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            nn.init.trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = self.fc1(x)
        x = self.conv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


