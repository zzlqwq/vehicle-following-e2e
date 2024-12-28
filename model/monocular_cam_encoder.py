import torch.nn as nn
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear',
                                align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear',
                                align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear',
                                 align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.up_8(x1)
        x2 = self.up_4(x2)
        x3 = self.up_2(x3)
        result = self.conv(torch.cat([x1, x2, x3, x4], dim=1))

        return result


class CamEncode(nn.Module):
    def __init__(self, D, cfg):
        super(CamEncode, self).__init__()
        self.D = D
        self.cfg = cfg
        self.C = self.cfg['cam_encoder_in_channel']

        self.trunk = EfficientNet.from_pretrained(self.cfg['backbone'])
        if self.cfg['backbone'] == 'efficientnet-b0':
            self.up1_in_channels = 320 + 112 + 40 + 24
        elif self.cfg['backbone'] == 'efficientnet-b4':
            self.up1_in_channels = 448 + 160

        self.up1 = Up(self.up1_in_channels, 512)
        self.depth_net = nn.Conv2d(512, self.D, kernel_size=1, padding=0)
        self.semantic_net = nn.Conv2d(512, 2, kernel_size=1, padding=0)
        self.feature_net = nn.Conv2d(512, self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):

        x = self.get_eff_depth(x)

        batch_size, _, h, w = x.shape

        # Depth
        depth = self.depth_net(x)
        depth = self.get_depth_dist(depth)

        # Semantic
        semantic_mask = self.semantic_net(x)
        # binary semantic
        semantic_mask = semantic_mask.softmax(dim=1)

        # Feature
        feature = self.feature_net(x)

        # Calculate new feature only for the foreground
        new_feature = depth.unsqueeze(1) * feature.unsqueeze(2)

        return depth, new_feature, semantic_mask

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'],
                     endpoints['reduction_2'], endpoints['reduction_1'])
        return x

    def forward(self, x):
        depth, x, semantic_mask = self.get_depth_feat(x)

        return x, depth, semantic_mask
