import inspect
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from mmcv.cnn.bricks import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.registry import MODELS
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn  # , set_deterministic
from torch.types import Device, _size
from torchonn.op.mrr_op import *

__all__ = [
    "LinearBlock",
    "ConvBlock",
    "TeMPO_Base",
]


def build_linear_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type="Linear")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `linear_layer` cannot be found
    # in the registry, fallback to search `linear_layer` in the
    # mmengine.MODELS.
    with MODELS.switch_scope_and_registry(None) as registry:
        linear_layer = registry.get(layer_type)
    if linear_layer is None:
        raise KeyError(
            f"Cannot find {linear_layer} in registry under scope "
            f"name {registry.scope}"
        )
    layer = linear_layer(*args, **kwargs, **cfg_)

    return layer


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        linear_cfg: dict = dict(type="TeMPOBlockLinear"),
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        dropout: float = 0.0,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
        if linear_cfg["type"] not in {"Linear", None}:
            linear_cfg.update({"device": device})
        self.linear = build_linear_layer(
            linear_cfg,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_features)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        bias: bool = False,
        conv_cfg: dict = dict(type="TeMPOBlockConv2d"),
        norm_cfg: dict | None = dict(type="BN", affine=True),
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        super().__init__()
        conv_cfg = conv_cfg.copy()
        if conv_cfg["type"] not in {"Conv2d", None}:
            conv_cfg.update({"device": device})
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm_cfg is not None:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TeMPO_Base(nn.Module):
    def __init__(
        self,
        *args,
        conv_cfg=dict(type="TeMPOBlockConv2d"),
        linear_cfg=dict(type="TeMPOBlockLinear"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with MODELS.switch_scope_and_registry(None) as registry:
            self._conv = (registry.get(conv_cfg["type"]),)
            self._linear = (registry.get(linear_cfg["type"]),)
            self._conv_linear = self._conv + self._linear

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._conv_linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_phase_variation(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_phase_variation"
            ):
                layer.set_phase_variation(flag)

    def set_output_noise(self, noise_std: float = 0.0):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_output_noise"
            ):
                layer.set_output_noise(noise_std)

    def set_light_redist(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_light_redist"
            ):
                layer.set_light_redist(flag)

    def set_input_power_gating(self, flag: bool = True, ER: float = 6) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_input_power_gating"
            ):
                layer.set_input_power_gating(flag, ER)

    def set_output_power_gating(self, flag: bool = True) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_output_power_gating"
            ):
                layer.set_output_power_gating(flag)

    def set_gamma_noise(
        self, noise_std: float = 0.0, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_gamma_noise"
            ):
                layer.set_gamma_noise(noise_std, random_state=random_state)

    def set_crosstalk_noise(
        self,
        flag: bool = True,
        first_conv_layer: bool = True,  # add crosstalk to first conv
        last_linear_layer: bool = False,  ## distable crosstalk for last linear
    ) -> None:
        modules = [m for m in self.modules() if isinstance(m, self._conv_linear)]
        if not first_conv_layer:
            for i, layer in enumerate(modules):
                if isinstance(layer, self._conv):
                    break
            modules.pop(i)
        if not last_linear_layer:
            for i, layer in enumerate(modules[::-1]):
                if isinstance(layer, self._linear):
                    break
            modules.pop(len(modules) - 1 - i)

        for layer in modules:
            if hasattr(layer, "set_crosstalk_noise"):
                layer.set_crosstalk_noise(flag)

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_noise"
            ):
                layer.set_weight_noise(noise_std)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_bitwidth"
            ):
                layer.set_weight_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_input_bitwidth"
            ):
                layer.set_input_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Dict[str, Tensor]]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for layer_name, layer_param_dict in param_dict.items():
            self.layers[layer_name].load_parameters(layer_param_dict)

    def build_obj_fn(self, X: Tensor, y: Tensor, criterion: Callable) -> Callable:
        def obj_fn(X_cur=None, y_cur=None, param_dict=None):
            if param_dict is not None:
                self.load_parameters(param_dict)
            if X_cur is None or y_cur is None:
                data, target = X, y
            else:
                data, target = X_cur, y_cur
            pred = self.forward(data)
            return criterion(pred, target)

        return obj_fn

    def enable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "enable_fast_forward"
            ):
                layer.enable_fast_forward()

    def disable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "disable_fast_forward"
            ):
                layer.disable_fast_forward()

    def sync_parameters(self, src: str = "weight") -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "sync_parameters"
            ):
                layer.sync_parameters(src=src)

    def build_weight(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(layer, "build_weight"):
                layer.build_weight()

    def print_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "print_parameters"
            ):
                layer.print_parameters()

    def set_noise_schedulers(
        self,
        scheduler_dict={
            "phase_variation_scheduler": None,
            "crosstalk_scheduler": None,
        },
    ):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                for scheduler_name, scheduler in scheduler_dict.items():
                    setattr(layer, scheduler_name, scheduler)

        for scheduler_name, scheduler in scheduler_dict.items():
            setattr(self, scheduler_name, scheduler)

    def reset_noise_schedulers(self):
        self.phase_variation_scheduler.reset()
        self.crosstalk_scheduler.reset()

    def step_noise_scheduler(self, T=1):
        if self.phase_variation_scheduler is not None:
            for _ in range(T):
                self.phase_variation_scheduler.step()

    def cycles(self, x_size, R: int = 8, C: int = 8) -> float:
        x = torch.randn(x_size, device=self.device)
        self.eval()

        def hook(m, inp):
            m._input_shape = inp[0].shape

        handles = []
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                handle = layer.register_forward_pre_hook(hook)
                handles.append(handle)
        with torch.no_grad():
            self.forward(x)
        cycles = {}  # name: (cycles_per_block, total_cycles)
        for name, layer in self.named_modules():
            if isinstance(layer, self._conv_linear):
                cycles[name] = layer.cycles(layer._input_shape, R=R, C=C)
        for handle in handles:
            handle.remove()
        return np.sum(list(cycles.values())), cycles

    def set_enable_ste(self, enable_ste: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._enable_ste = enable_ste

    def set_noise_flag(self, noise_flag: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._noise_flag = noise_flag

    def calc_weight_MZI_energy(
        self,
        input_size=[1, 3, 32, 32],
        R: int = 8,
        C: int = 8,
        freq: float = 1.0,  # GHz
    ) -> None:
        ## return total energy in mJ and power mW breakdown

        total_cycles, cycle_dict = self.cycles(input_size, R=R, C=C)
        power_dict = {}
        energy_dict = {}
        with torch.no_grad():
            for name, layer in self.named_modules():
                if isinstance(layer, self._conv_linear) and hasattr(
                    layer, "calc_weight_MZI_power"
                ):
                    power = layer.calc_weight_MZI_power(
                        src="weight", reduction="none"
                    )  # [p,q,r,c,k1,k1] -> 1 # mW

                    ## calculate energy
                    ## (P1*cyc_per_clk + P2*cyc_per_clk + ... + P_{RC} * cyc_per_clk) / freq
                    ## (P1+P2+P3+...+P_{RC}) * cyc_per_clock / freq
                    ## sum(P) * cyc_per_clock / freq
                    power = power.sum().item()
                    power_dict[name] = power
                    cycles_per_block = cycle_dict[name][0]
                    energy_dict[name] = power * cycles_per_block / freq / 1e9  # mJ
        total_energy = np.sum(list(energy_dict.values()))
        avg_power = total_energy / (total_cycles / freq / 1e9)
        return (
            total_energy,  # mJ
            energy_dict,  # layer-wise energy breakdown
            total_cycles,  # total cycles
            cycle_dict,  # layer-wise cycle breakdown
            avg_power,  # average power mW
            power_dict,  # layer-wise power breakdown
        )

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
