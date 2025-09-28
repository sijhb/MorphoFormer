#!/usr/bin/python
# author htkk1111
# 2025年02月19日


import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from models.df import Dilateformer
from models.CrossAttentionBlock import InceptionBottleneck



class DualDilateformer(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=[8, 16], in_chans=3, num_classes=1000, embed_dim=(72, 96),
                 pretrained=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs):
        super().__init__()

        self.img_size = img_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # 初始化两个分支
        self.branch1 = Dilateformer(img_size=img_size[0], patch_size=patch_size[0],
                                    depths=[2, 2, 6, 2], embed_dim=embed_dim[0], num_heads=[3, 6, 12, 24],  **kwargs)
        self.branch1.default_cfg = _cfg()

        self.branch2 = Dilateformer(img_size=img_size[1], patch_size=patch_size[1],
                                    depths=[2, 2, 6, 2], embed_dim=embed_dim[1], num_heads=[3, 6, 12, 24],  **kwargs)
        self.branch2.default_cfg = _cfg()

        self.head = InceptionBottleneck(in_channels=1344, out_channels=576, kernel_sizes=(3, 5, 7, 9, 11))

        self.norm = norm_layer(576)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(576, num_classes) if num_classes > 0 else nn.Identity()


    def forward(self, x):


        B, C, H, W = x.shape
        xs = []
        for i in range(2):  # 第一个阶段
            x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
            xs.append(x_)

        # 获取每个分支的输出
        out1 = self.branch1(xs[0])
        out2 = self.branch2(xs[1])

        out = self.head([out1, out2])

        out = out.flatten(2).transpose(1, 2)
        out = self.norm(out)  # B L C
        out = self.avgpool(out.transpose(1, 2))  # B C 1
        out = torch.flatten(out, 1)


        # 最终输出
        return self.out(out)



if __name__ == '__main__':
    model = DualDilateformer()
    x = torch.rand([1, 3, 224, 224])
    y = model(x)
    print(y)



