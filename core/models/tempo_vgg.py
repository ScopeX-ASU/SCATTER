import torch
from torch import Tensor, nn
from torch.types import Device

from core.models.layers.utils import *

from .tempo_base import ConvBlock, LinearBlock, TeMPO_Base

__all__ = [
    "TeMPO_VGG8",
    "TeMPO_VGG11",
    "TeMPO_VGG13",
    "TeMPO_VGG16",
    "TeMPO_VGG19",
    "VGG8",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
]

cfg_32 = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

cfg_64 = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "GAP"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "GAP"],
    "vgg13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "GAP",
    ],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "GAP",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "GAP",
    ],
}


class VGG(TeMPO_Base):
    """MRR VGG (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication."""

    def __init__(
        self,
        vgg_name: str,
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
        super().__init__()

        self.vgg_name = vgg_name
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
        cfg = cfg_32 if self.img_height == 32 else cfg_64
        self.features, convNum = self._make_layers(cfg[self.vgg_name])
        # build FC layers
        ## lienar layer use the last miniblock
        if (
            self.img_height == 64 and self.vgg_name == "vgg8"
        ):  ## model is too small, do not use dropout
            classifier = []
        else:
            classifier = [nn.Dropout(0.5)]
        classifier += [
            LinearBlock(
                512,
                self.num_classes,
                bias=False,
                linear_cfg=self.linear_cfg,
                act_cfg=None,
                device=self.device,
            )
        ]
        self.classifier = nn.Sequential(*classifier)

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        convNum = 0

        for x in cfg:
            # MaxPool2d
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == "GAP":
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
            else:
                # conv + BN + RELU
                layers += [
                    ConvBlock(
                        in_channels,
                        x,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        device=self.device,
                    )
                ]
                in_channels = x
                convNum += 1
        return nn.Sequential(*layers), convNum

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def TeMPO_VGG8(*args, **kwargs):
    return VGG("vgg8", *args, **kwargs)


def TeMPO_VGG11(*args, **kwargs):
    return VGG("vgg11", *args, **kwargs)


def TeMPO_VGG13(*args, **kwargs):
    return VGG("vgg13", *args, **kwargs)


def TeMPO_VGG16(*args, **kwargs):
    return VGG("vgg16", *args, **kwargs)


def TeMPO_VGG19(*args, **kwargs):
    return VGG("vgg19", *args, **kwargs)


def VGG8(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return VGG("vgg8", *args, **kwargs)


def VGG11(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return VGG("vgg11", *args, **kwargs)


def VGG13(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return VGG("vgg13", *args, **kwargs)


def VGG16(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return VGG("vgg16", *args, **kwargs)


def VGG19(*args, **kwargs):
    kwargs.pop("conv_cfg")
    kwargs.pop("linear_cfg")
    kwargs.update(dict(conv_cfg=dict(type="Conv2d"), linear_cfg=dict(type="Linear")))
    return VGG("vgg19", *args, **kwargs)
