from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.activation import ReLU
from torch.types import Device, _size

from core.models.layers.utils import *

from .tempo_base import ConvBlock, LinearBlock, TeMPO_Base

__all__ = [
    "TeMPO_ResNet18",
    "TeMPO_ResNet20",
    "TeMPO_ResNet32",
    "TeMPO_ResNet34",
    "TeMPO_ResNet50",
    "TeMPO_ResNet101",
    "TeMPO_ResNet152",
    "ResNet18",
    "ResNet20",
    "ResNet32",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]


def conv3x3(
    in_planes,
    out_planes,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    device: Device = torch.device("cuda"),
    conv_cfg: dict = dict(type="TeMPOBlockConv2d"),
):
    conv = ConvBlock(
        in_planes,
        out_planes,
        3,
        bias=bias,
        stride=stride,
        padding=padding,
        conv_cfg=conv_cfg,
        act_cfg=None,
        norm_cfg=None,
        device=device,
    )

    return conv


def conv1x1(
    in_planes,
    out_planes,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    device: Device = torch.device("cuda"),
    conv_cfg: dict = dict(type="TeMPOBlockConv2d"),
):
    conv = ConvBlock(
        in_planes,
        out_planes,
        1,
        bias=bias,
        stride=stride,
        padding=padding,
        conv_cfg=conv_cfg,
        act_cfg=None,
        norm_cfg=None,
        device=device,
    )

    return conv


def Linear(
    in_features,
    out_features,
    bias: bool = False,
    device: Device = torch.device("cuda"),
    linear_cfg: dict = dict(type="TeMPOBlockLinear"),
):
    linear = LinearBlock(
        in_features,
        out_features,
        bias=bias,
        linear_cfg=linear_cfg,
        norm_cfg=None,
        act_cfg=None,
        dropout=0.0,
        device=device,
    )

    return linear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        device: Device = torch.device("cuda"),
        conv_cfg: dict = dict(type="TeMPOBlockConv2d"),
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            in_planes,
            planes,
            bias=False,
            stride=stride,
            padding=1,
            device=device,
            conv_cfg=conv_cfg,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = ReLU(inplace=True)
        self.conv2 = conv3x3(
            planes,
            planes,
            bias=False,
            stride=1,
            padding=1,
            device=device,
            conv_cfg=conv_cfg,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = ReLU(inplace=True)

        self.shortcut = nn.Identity()
        # self.shortcut.conv1_spatial_sparsity = self.conv1.bp_input_sampler.spatial_sparsity
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    bias=False,
                    stride=stride,
                    padding=0,
                    device=device,
                    conv_cfg=conv_cfg,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        device: Device = torch.device("cuda"),
        conv_cfg: dict = dict(type="TeMPOBlockConv2d"),
    ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(
            in_planes,
            planes,
            bias=False,
            stride=1,
            padding=0,
            device=device,
            conv_cfg=conv_cfg,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = ReLU(inplace=True)
        self.conv2 = conv3x3(
            planes,
            planes,
            bias=False,
            stride=stride,
            padding=1,
            device=device,
            conv_cfg=conv_cfg,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = ReLU(inplace=True)
        self.conv3 = conv1x1(
            planes,
            self.expansion * planes,
            bias=False,
            stride=1,
            padding=0,
            device=device,
            conv_cfg=conv_cfg,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.act3 = ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    bias=False,
                    stride=stride,
                    padding=0,
                    device=device,
                    conv_cfg=conv_cfg,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(TeMPO_Base):
    """MRR ResNet (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication."""

    def __init__(
        self,
        block,
        num_blocks,
        in_planes,
        img_height: int,
        img_width: int,
        in_channels: int,
        num_classes: int,
        conv_cfg=dict(type="TeMPOBlockConv2d"),
        linear_cfg=dict(type="TeMPOBlockLinear"),
        norm_cfg=dict(type="BN", affine=True),
        act_cfg=dict(type="ReLU", inplace=True),
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__(conv_cfg=conv_cfg, linear_cfg=linear_cfg)

        # resnet params
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.img_height = img_height
        self.img_width = img_width

        self.in_channels = in_channels
        self.num_classes = num_classes

        # list of block size
        self.conv_cfg = conv_cfg
        self.linear_cfg = linear_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.device = device

        # build layers
        blkIdx = 0
        self.conv1 = conv3x3(
            in_channels,
            self.in_planes,
            bias=False,
            stride=1 if img_height <= 64 else 2,  # downsample for imagenet, dogs, cars
            padding=1,
            device=self.device,
            conv_cfg=self.conv_cfg,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        blkIdx += 1

        self.layer1 = self._make_layer(
            block,
            in_planes,
            num_blocks[0],
            stride=1,
            device=device,
            conv_cfg=self.conv_cfg,
        )
        blkIdx += 1

        self.layer2 = self._make_layer(
            block,
            in_planes * 2,
            num_blocks[1],
            stride=2,
            device=device,
            conv_cfg=self.conv_cfg,
        )
        blkIdx += 1

        self.layer3 = self._make_layer(
            block,
            in_planes * 4,
            num_blocks[2],
            stride=2,
            device=device,
            conv_cfg=self.conv_cfg,
        )
        blkIdx += 1

        self.layer4 = self._make_layer(
            block,
            in_planes * 8,
            num_blocks[3],
            stride=2,
            device=device,
            conv_cfg=self.conv_cfg,
        )
        blkIdx += 1

        n_channel = in_planes * 8 if num_blocks[3] > 0 else in_planes * 4
        self.linear = Linear(
            n_channel * block.expansion,
            self.num_classes,
            bias=False,
            device=device,
            linear_cfg=self.linear_cfg,
        )

        self.drop_masks = None

        self.reset_parameters()
        self.set_phase_variation(False)
        self.set_crosstalk_noise(False)
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0
        self.set_weight_noise(0.0)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        device: Device = torch.device("cuda"),
        conv_cfg: dict = dict(type="TeMPOBlockConv2d"),
    ):
        if num_blocks == 0:
            return nn.Identity()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    device=device,
                    conv_cfg=conv_cfg,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if x.size(-1) > 64:  # 224 x 224, e.g., cars, dogs, imagenet
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def TeMPO_ResNet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], 64, *args, **kwargs)


def TeMPO_ResNet20(*args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3, 0], 16, *args, **kwargs)


def TeMPO_ResNet32(*args, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5, 0], 16, *args, **kwargs)


def TeMPO_ResNet34(*args, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], 64, *args, **kwargs)


def TeMPO_ResNet50(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], 64, *args, **kwargs)


def TeMPO_ResNet101(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], 64, *args, **kwargs)


def TeMPO_ResNet152(*args, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], 64, *args, **kwargs)


def ResNet18(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return ResNet(BasicBlock, [2, 2, 2, 2], 64, *args, **kwargs)


def ResNet20(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return ResNet(BasicBlock, [3, 3, 3, 0], 16, *args, **kwargs)


def ResNet32(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return ResNet(BasicBlock, [5, 5, 5, 0], 16, *args, **kwargs)


def ResNet34(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return ResNet(BasicBlock, [3, 4, 6, 3], 64, *args, **kwargs)


def ResNet50(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return ResNet(Bottleneck, [3, 4, 6, 3], 64, *args, **kwargs)


def ResNet101(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return ResNet(Bottleneck, [3, 4, 23, 3], 64, *args, **kwargs)


def ResNet152(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return ResNet(Bottleneck, [3, 8, 36, 3], 64, *args, **kwargs)
