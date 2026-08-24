import torch
import torch.nn as nn

from mmcv.models.bricks import (
    ConvModule,
    Linear,
    build_activation_layer,
    build_conv_layer,
    build_norm_layer,
    build_plugin_layer,
)
from mmcv.models.utils import (
    bias_init_with_prob,
    constant_init,
    fuse_conv_bn,
    xavier_init,
)


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


__all__ = [
    "ConvModule",
    "Linear",
    "Scale",
    "bias_init_with_prob",
    "build_activation_layer",
    "build_conv_layer",
    "build_norm_layer",
    "build_plugin_layer",
    "constant_init",
    "fuse_conv_bn",
    "xavier_init",
]
