import torch
import time
import numpy as np

from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES
import math

FLOPS_COUNTER = 0

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad,
                                             dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1)**2
        FLOPS_COUNTER += a * b * output_points * (ks**2)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5

        m = torch.nn.Conv2d(w.size(1), w.size(0),
                            w.shape[2:], stride=self.c.stride, padding=self.c.padding,
                            dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))

        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution**2
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()

        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5

        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x, H=None, W=None):
        start = time.time()
        l, bn = self._modules.values()
        x = l(x)
        H, W = x.shape[1], x.shape[2]
        dua = time.time()-start
        #print('linear dua', dua)
        return bn(x.flatten(0, 1)).reshape_as(x)

class Residual(torch.nn.Module):
    def __init__(self, m, drop, mr=0):
        super().__init__()
        self.m = m
        self.drop = drop
        self.mr = mr

    def forward(self, x, H=None, W=None):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class Channel_via_Residual(torch.nn.Module):
    def __init__(self, m, drop, mr=0):
        super().__init__()
        self.m = m
        self.drop = drop
        self.mr = mr

    def forward(self, x, last_q, H=None, W=None):
        if self.training and self.drop > 0:
            return x + self.m(x, last_q) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            out_x = self.m(x, last_q)
            return x+out_x

class Channel_via_MSA(torch.nn.Module):
    def __init__(self, dim, out_dim, dim_ratio=1, num_heads=8, qkv_bias=False,
                 activation=None, attn_drop=0., proj_drop=0.):
        # last_shape: the shape of last query/attn
        super().__init__()
        assert np.mod((dim * dim_ratio), num_heads) == 0, \
               '*** in Channel_via_MSA, mod(self.dim_ratio * self.dim, self.num_heads) != 0'
        self.num_heads = num_heads
        self.temperature = torch.nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = torch.nn.Linear(dim, dim * dim_ratio * 3, bias=qkv_bias)
        self.dim_ratio = dim_ratio
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.activation = activation()
        self.proj = torch.nn.Linear(dim * dim_ratio, out_dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim_ratio * C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        out_attn = (q @ k.transpose(-2, -1)) * self.temperature
        v = v.transpose(-2, -1)
        attn = self.attn_drop(out_attn.softmax(dim=-1))
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, -1)
        x = self.activation(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class Spatial_via_Conv(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 kernel_size=3, act_layer=torch.nn.GELU,
                 depth=2, residual_block=False, drop_path=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.block = None
        sc_conv1 = torch.nn.Conv2d(in_features, in_features,
                                     kernel_size = kernel_size, stride = 1, padding = int(kernel_size // 2),
                                     groups = in_features, bias = False)
        sc_act = act_layer()
        sc_bn = torch.nn.SyncBatchNorm(in_features)
        sc_conv2 = torch.nn.Conv2d(in_features, in_features, kernel_size = kernel_size,
                                     stride = 1, padding = int(kernel_size // 2),
                                     groups = in_features, bias = False)
        if in_features == out_features:
            self.block = torch.nn.Sequential(sc_conv1, sc_act, sc_bn, sc_conv2,)
        else:
            self.block = torch.nn.Sequential(sc_conv1, sc_act, sc_bn,
                                             sc_conv2, sc_act, torch.nn.SyncBatchNorm(in_features),
                                             torch.nn.Conv2d(in_features, out_features, kernel_size = 1, bias = False))

        if residual_block:
            self.block = Residual(self.block, drop_path)

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        x_2d = x.transpose(-2, -1).reshape(B, C, H, W)
        out = self.block(x_2d)
        return out.flatten(2).transpose(-2, -1)


