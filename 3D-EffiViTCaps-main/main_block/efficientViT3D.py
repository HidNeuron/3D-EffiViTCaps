# --------------------------------------------------------
# EfficientViT3D Model Block
# Copyright (c) 2024 UESTC
# Build the EfficientViT3D Model
# Written by: Dongwei Gan
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, SEModule3D


class Conv3d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv3d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm3d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv3d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv3d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv3d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention3D(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, key_dim, num_heads=4,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv3d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv3d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(),
            Conv3d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution),)
        points = list(itertools.product(range(resolution), range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W,D)
        B, C, H, W, D = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W, D).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W, D
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
            attn = (
                    (q.transpose(-2, -1) @ k) * self.scale
                    +
                    (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1)  # B N N
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W, D)  # B C H W D
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention3D(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, key_dim, num_heads=4,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention3D(dim, key_dim, num_heads,
                                           attn_ratio=attn_ratio,
                                           resolution=window_resolution,
                                           kernels=kernels, )

    def forward(self, x):
        H = W = D = self.resolution
        B, C, H_, W_, D_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_ and D == D_, 'input feature has wrong size, expect {}, got {}'.format((H, W, D), (H_, W_, D_))

        if H <= self.window_resolution and W <= self.window_resolution and D <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            pad_h = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_w = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            pad_d = (self.window_resolution - D %
                     self.window_resolution) % self.window_resolution
            padding = pad_h > 0 or pad_w > 0 or pad_d > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))

            pH, pW , pD = H + pad_h, W + pad_w, D + pad_d
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            nD = pD // self.window_resolution
            # window partition, B H W D C -> B nH h nW w nD d C -> B nH h nW w nD d C -> B*nH*nW*nD h w d C -> B*nH*nW*nD C h w d
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, nD, self.window_resolution, C).\
                permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
                B * nH * nW * nD, self.window_resolution, self.window_resolution, self.window_resolution, C
            ).permute(0, 4, 1, 2, 3)
            x = self.attn(x)
            # window reverse, B*nH*nW*nD C h w d -> B*nH*nW*nD h w d C -> B nH nW nD h w d C -> B nH h nW w nD d C -> B H W D C
            x = x.permute(0, 2, 3, 4, 1).view(B, nH, nW, nD, self.window_resolution, self.window_resolution,
            self.window_resolution, C).permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, pH, pW, pD, C)
            if padding:
                x = x[:, :H, :W, :D].contiguous()
            x = x.permute(0, 4, 1, 2, 3)
        return x


class EfficientViTBlock3D(torch.nn.Module):
    """ A basic 3D EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, type,
                 ed, kd, nh=4,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()

        self.dw0 = Residual(Conv3d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention3D(ed, kd, nh, attn_ratio=ar, \
                                                       resolution=resolution, window_resolution=window_resolution,
                                                       kernels=kernels))

        self.dw1 = Residual(Conv3d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class PatchMerging3D(nn.Module):
    """ 3D Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_dim, output_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(8 * input_dim, output_dim, bias=False)
        self.norm = norm_layer(8 * input_dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, H, W, D).
        """
        x = x.transpose(1, 4)
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (D % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, D % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x3 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x4 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x6 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.transpose(1, 4)
        return x


class PatchExpand3D(nn.Module):
    """ 3D Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_dim, output_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(input_dim, output_dim * 8, bias=False)
        self.norm = norm_layer(output_dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, H, W, D).
        """
        x = x.transpose(1, 4)
        B, D, H, W, C = x.shape

        x = self.expand(x)
        # assert L == D * H * W, "input feature has wrong size"

        x = x.view(B, D * 2, H * 2, W * 2, -1)
        x = self.norm(x)
        x = x.transpose(1, 4)

        return x
