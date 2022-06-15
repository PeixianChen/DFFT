import torch
import torch.nn as nn
import torch.nn.functional as F

#from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0., kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)
        self.act = act_layer()
        self.bn = nn.SyncBatchNorm(in_features)
        self.conv2 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


class CABlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., group=-1, reducedim=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.reducedim = reducedim
        if reducedim:
            #if group > 0:
            #    self.qkv = nn.Conv1d(dim, dim*2, 1, groups=group, bias=qkv_bias)
            #else:
            self.qkv = nn.Linear(dim, dim*2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.group = group
        if self.group < 0:
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj = nn.Conv1d(dim, dim, 1, groups=group)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.reducedim:
            #if self.group > 0:
            #    qkv = self.qkv(x.permute(0,2,1).contiguous()).permute(0,2,1).contiguous().reshape(B, N, 2, self.num_heads, C // self.num_heads)
            #else:
            qkv = self.qkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[0]
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        if self.group < 0:
            x = self.proj(x)
        else:
            x = x.permute(0, 2, 1).contiguous()
            x = self.proj(x)
            x = x.permute(0, 2, 1).contiguous()
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class CMlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., group=-1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group > 0:
            self.fc1 = nn.Conv1d(in_features, hidden_features, 1, groups=group)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if group > 0:
            self.fc2 = nn.Conv1d(hidden_features, out_features, 1, groups=group)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.group = group

    def forward(self, x):
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        return x




class GroupCA(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_tokens=196, eta=None, lmlp=True, ffnmlp=True, cmlp=False, normlatter=False, group=-1,
                 reducedim=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CABlock(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, group=group, reducedim=reducedim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop, group=group)

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        self.lmlp = lmlp
        self.ffnmlp = ffnmlp
        self.cmlp = cmlp
        self.normlatter = normlatter
        if self.cmlp:
            if dim==256:
                self.norm4 = norm_layer(384)   #norm_layer(dim//2)
                self.cmlp_layer = CMlp(384, dim)    #CMlp(dim//2, dim)
            else:
                self.norm4 = norm_layer(dim//2)
                self.cmlp_layer = CMlp(dim//2, dim)

        if self.normlatter:
            self.norml = norm_layer(dim)

    def forward(self, x, H, W):
        if self.cmlp:
            x = self.cmlp_layer(self.norm4(x))

        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.lmlp:
            x = x + self.drop_path(self.local_mp(self.norm3(x), H, W))
        if self.ffnmlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.normlatter:
            x = self.norml(x)
        return x



