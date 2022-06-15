# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

from numpy.core.numeric import cross
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES
#from .swin_utils import CSP_DenseBlock
from .CA_layer import *
from .SA_layer import *
from .DOT_blocks import *


def b16(n, activation, resolution=224):
    #  Conv2d_BN(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000)
    return torch.nn.Sequential(
        Conv2d_BN(3, n, 3, 2, 1, resolution=resolution),
        activation(),)
        #Conv2d_BN(n // 2, n, 3, 1, 1, resolution=resolution),
        #activation(),)
        #Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        #activation(),
        #Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))

@BACKBONES.register_module()
class DFFTNet(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 alldepths=[3, 3, 19, 3],
                 num_heads=[4, 4, 7, 12],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 2, 4, 6),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 crossca_position = [1, 2, 3], 
                 crossca_type = "CrossAddCa_a_n_l"):
        super().__init__()

        print("depths:", depths)
        print("num_heads", num_heads)
        print("crossca_position:", crossca_position)

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = num_heads[0] * 32
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.b16 = b16(self.embed_dim, torch.nn.Hardswish)
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=self.embed_dim, embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(num_heads[i_layer]*32)
            layer_dimout = int(num_heads[i_layer+1]*32) if (i_layer < self.num_layers - 1) else int(num_heads[i_layer]*32)
            layer = DOTBlock(
                dim=layer_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                alldepths=alldepths[i_layer])
            self.layers.append(layer)
            if i_layer in crossca_position:
                saablock = SAAModule(layer_dim, 2, torch.nn.Hardswish, resolution=224, drop_path=0.,)
                self.layers.append(saablock)
            if (i_layer < self.num_layers - 1):
                downsample = PatchMerging(dim=layer_dim, dimout = layer_dimout, norm_layer=norm_layer)
                self.layers.append(downsample)

        num_features = [int(num_heads[i]*32) for i in range(self.num_layers)]
        self.num_features = num_features
        self.upsample_2 = torch.nn.Upsample(scale_factor=2)

        self.links = nn.ModuleList()
        for i_layer in range(1, self.num_layers):
            if i_layer == 1:
                layer_dim = 128
            else:
                layer_dim = self.num_features[-1]
            saeblock = SAEBlock(layer_dim, 2, torch.nn.Hardswish, resolution=224, drop_path=0.)
            self.links.append(saeblock)

        self.out_norm = norm_layer(self.num_features[-1])

        saaconv = []
        saaconv.append(nn.Sequential(nn.Conv2d(self.num_features[0], self.num_features[1], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[1]), nn.ReLU6(inplace=True)))
        saaconv.append(nn.Sequential(nn.Conv2d(self.num_features[1], self.num_features[1], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[1]), nn.ReLU6(inplace=True)))
        saaconv.append(nn.Sequential(nn.Conv2d(self.num_features[1], self.num_features[2], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[2]), nn.ReLU6(inplace=True)))
        saaconv.append(nn.Sequential(nn.Conv2d(self.num_features[2], self.num_features[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[3]), nn.ReLU6(inplace=True)))
        for idx in range(4):
            layer_name = f'saaconv{idx}'
            self.add_module(layer_name, saaconv[idx])   

        saeconv = []
        saeconv = []
        saeconv.append(nn.Sequential(nn.Conv2d(self.num_features[1], 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU6(inplace=True)))
        saeconv.append(nn.Sequential(nn.Conv2d(self.num_features[2], 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU6(inplace=True)))
        saeconv.append(nn.Sequential(nn.Conv2d(128, self.num_features[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[3]), nn.ReLU6(inplace=True)))
        saeconv.append(nn.Sequential(nn.Conv2d(self.num_features[3], self.num_features[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[3]), nn.ReLU6(inplace=True)))
        for idx in range(4):
            layer_name = f'saeconv{idx}'
            self.add_module(layer_name, saeconv[idx])   

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.b16(x)
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        dot_feature, dot_HW = [], []
        saa_feature = []
        channel = [self.num_features[1], self.num_features[1], self.num_features[2], self.num_features[3]]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DOTBlock):
                x, H, W = layer(x, Wh, Ww)
                ca_x = x
                B, _, C = x.shape
                cross_x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                conv_layer = getattr(self, f'saaconv{len(dot_feature)}')
                if len(dot_feature) < 2:
                    cross_x = conv_layer(cross_x)
                dot_feature.append(cross_x.contiguous().view(B, channel[len(dot_feature)], -1).transpose(-2, -1))
                dot_HW.append([H, W])
            elif isinstance(layer, SAAModule):
                if i == len(self.layers)-1:
                    last_layer = True
                    x, _ = layer(dot_feature[-2:], dot_HW[-2:], last_layer=last_layer)
                    saa_feature.append(x)
                    ca_x = x
                else:
                    link_x, x = layer(dot_feature[-2:], dot_HW[-2:])
                    saa_feature.append(link_x)
                    if len(dot_feature) > 1:
                        cross_x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                        conv_layer = getattr(self, f'saaconv{len(dot_feature)}')
                        cross_x = conv_layer(cross_x)
                        dot_feature[-1] =cross_x.contiguous().view(B, channel[len(dot_feature)], -1).transpose(-2, -1)
                    else:
                        dot_feature[-1] = link_x
            elif isinstance(layer, PatchMerging):
                x = layer(x, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
        
        saa_feature.append(dot_feature[-1])

        # addlink2:
        dot_feature = saa_feature

        channel = [128, 128, self.num_features[3], self.num_features[3]]
        for i in range(2):
            H, W = dot_HW[i]
            B, _, C = dot_feature[i].shape
            cross_x = dot_feature[i].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            conv_layer = getattr(self, f'saeconv{i}')
            cross_x = conv_layer(cross_x)
            dot_feature[i] = cross_x.contiguous().view(B, channel[i], -1).transpose(-2, -1)
        for i in range(len(self.links)):
            H, W = dot_HW[i+1]
            B, _, C = dot_feature[i+1].shape
            layer = self.links[i]
            if i == len(self.links)-1:
                last_layer = True
                x, _ = layer(dot_feature[i:i+2], dot_HW[i:i+2], last_layer=last_layer)
            else:
                conv_layer = getattr(self, f'saeconv{i+2}')
                last_layer = False
                _, x = layer(dot_feature[i:i+2], dot_HW[i:i+2], last_layer=last_layer)
                x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                x = conv_layer(x)
                dot_feature[i+1] = x.contiguous().view(B, channel[i+2], -1).transpose(-2, -1)

        x = self.out_norm(x)
        x = x.view(-1, dot_HW[2][0], dot_HW[2][1], self.num_features[-1]).permute(0, 3, 1, 2).contiguous()
        return tuple([x])

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DFFTNet, self).train(mode)
        self._freeze_stages()
# 
