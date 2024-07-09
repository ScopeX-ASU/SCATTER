from collections import OrderedDict
from typing import List

import torch
from torch import Tensor, nn
from torch.types import Device

from core.models.layers.utils import *

from .tempo_base import ConvBlock, LinearBlock, TeMPO_Base

__all__ = ["TeMPO_CNN", "CNN"]


class TeMPO_CNN(TeMPO_Base):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channels: int,
        num_classes: int,
        kernel_list: List[int] = [32],
        kernel_size_list: List[int] = [3],
        stride_list=[1],
        padding_list=[1],
        dilation_list=[1],
        groups=1,
        pool_out_size: int = 5,
        hidden_list: List[int] = [32],
        conv_cfg=dict(type="TeMPOBlockConv2d"),
        linear_cfg=dict(type="TeMPOBlockLinear"),
        norm_cfg=dict(type="BN", affine=True),
        act_cfg=dict(type="ReLU", inplace=True),
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__(conv_cfg=conv_cfg, linear_cfg=linear_cfg)
        self.conv_cfg = conv_cfg
        self.linear_cfg = linear_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.groups = groups

        self.pool_out_size = pool_out_size
        self.hidden_list = hidden_list

        self.device = device

        self.build_layers()
        self.drop_masks = None

        self.reset_parameters()
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0

        self.set_phase_variation(False)
        self.set_crosstalk_noise(False)
        self.set_noise_schedulers()
        self.set_weight_noise(0.0)

    def build_layers(self):
        self.features = OrderedDict()
        for idx, out_channels in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx + 1)
            in_channels = self.in_channels if (idx == 0) else self.kernel_list[idx - 1]
            self.features[layer_name] = ConvBlock(
                in_channels,
                out_channels,
                self.kernel_size_list[idx],
                self.stride_list[idx],
                self.padding_list[idx],
                self.dilation_list[0],
                self.groups,
                bias=True,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,  # enable batchnorm
                act_cfg=self.act_cfg,  # enable relu
                device=self.device,
            )

        self.features = nn.Sequential(self.features)

        if self.pool_out_size > 0:
            self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)
            feature_size = (
                self.kernel_list[-1] * self.pool_out_size * self.pool_out_size
            )
        else:
            self.pool2d = None
            img_height, img_width = self.img_height, self.img_width
            for layer in self.modules():
                if isinstance(layer, self._conv):
                    img_height, img_width = layer.get_output_dim(img_height, img_width)
            feature_size = img_height * img_width * self.kernel_list[-1]

        self.classifier = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx + 1)
            in_features = feature_size if idx == 0 else self.hidden_list[idx - 1]
            out_features = hidden_dim
            self.classifier[layer_name] = LinearBlock(
                in_features,
                out_features,
                bias=True,
                linear_cfg=self.linear_cfg,
                act_cfg=self.act_cfg,
                device=self.device,
            )

        layer_name = "fc" + str(len(self.hidden_list) + 1)
        self.classifier[layer_name] = LinearBlock(
            self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
            self.num_classes,
            bias=True,
            linear_cfg=self.linear_cfg,
            act_cfg=None,
            device=self.device,
        )
        self.classifier = nn.Sequential(self.classifier)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.pool2d is not None:
            x = self.pool2d(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def CNN(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return TeMPO_CNN(*args, **kwargs)
