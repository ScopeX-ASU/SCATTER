from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pyutils.compute import add_gaussian_noise, gen_gaussian_noise
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.types import Device

from .utils import STE, mzi_out_diff_to_phase, mzi_phase_to_out_diff, partition_chunks

__all__ = ["ONNBaseLayer"]


class ONNBaseLayer(nn.Module):
    def __init__(self, *args, device: Device = torch.device("cpu"), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # cuda or cpu, defaults to cpu
        self.device = device

    def build_parameters(self, mode: str = "weight") -> None:
        ## weight mode
        phase = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        weight = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        # TIA gain
        S_scale = torch.ones(
            size=list(weight.shape[:-2]) + [1], device=self.device, dtype=torch.float32
        )

        if mode == "weight":
            self.weight = Parameter(weight)
        elif mode == "phase":
            self.phase = Parameter(phase)
            self.S_scale = Parameter(S_scale)
        else:
            raise NotImplementedError

        for p_name, p in {
            "weight": weight,
            "phase": phase,
            "S_scale": S_scale,
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    def reset_parameters(self, mode=None) -> None:
        mode = mode or self.mode
        if mode in {"weight"}:
            # init.kaiming_normal_(self.weight.data)
            if hasattr(self, "kernel_size"):  # for conv2d
                weight = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    bias=False,
                ).weight.data
                weight = weight.flatten(1)
                in_channels_pad = self.in_channels_pad - weight.shape[1]
                out_channels_pad = self.out_channels_pad - weight.shape[0]
            elif hasattr(self, "in_features"):  # for linear
                weight = nn.Linear(
                    self.in_features, self.out_features, bias=False
                ).weight.data
                in_channels_pad = self.in_features_pad - weight.shape[1]
                out_channels_pad = self.out_features_pad - weight.shape[0]
            weight = torch.nn.functional.pad(
                weight,
                (0, in_channels_pad, 0, out_channels_pad),
                mode="constant",
                value=0,
            )
            self.weight.data.copy_(
                partition_chunks(weight, out_shape=self.weight.shape).to(
                    self.weight.device
                )
            )

        elif mode in {"phase"}:
            self.reset_parameters(mode="weight")
            scale = self.weight.data.abs().flatten(-2, -1).max(dim=-1, keepdim=True)[0]
            self.S_scale.data.copy_(scale)
            self.phase.data.copy_(
                mzi_out_diff_to_phase(self.weight.data.div(scale[..., None]))
            )
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    @classmethod
    def from_layer(cls, layer: nn.Module, *args, **kwargs) -> nn.Module:
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_phase_variation(self, flag: bool = False) -> None:
        self._enable_phase_variation = flag

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        self.weight_noise_std = noise_std

    def set_output_noise(self, noise_std: float = 0.0) -> None:
        self.output_noise_std = noise_std

    def set_gamma_noise(
        self, noise_std: float, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std

    # crosstalk changes
    def set_crosstalk_noise(self, flag: bool = False) -> None:
        self._enable_crosstalk = flag

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_quantizer.set_bit(w_bit)
        self.weight_quantizer.set_bit(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bit(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def switch_mode_to(self, mode: str) -> None:
        self.mode = mode

    def set_enable_ste(self, enable_ste: bool) -> None:
        self._enable_ste = enable_ste

    def set_noise_flag(self, noise_flag: bool) -> None:
        self._noise_flag = noise_flag

    def _add_phase_variation(
        self, x, src: float = "weight", enable_remap: bool = False
    ) -> None:
        # Gaussian blur to process the noise distribution from
        # do not do inplace tensor modification to x, this is dynamic noise injection in every forward pass
        # this function can handle both phase noise injection to phase tensors and weight tensors
        if (not self._enable_phase_variation) or self.phase_variation_scheduler is None:
            return x  #  no noise injected

        if src == "weight":
            phase, S_scale = self.build_phase_from_weight_(x)
            # we need to use this func to update S_scale and keep track of S_scale if weight gets updated.
        elif src == "phase":
            phase = x
        else:
            raise NotImplementedError

        ## we need to remap noise distribution instead of phase or noises!
        # because multiple weights can mapto the same tile.
        # because even when they are mapped to the same tile, they cannot share noises, they share noise distribution.
        ## then we do not need to unapply_remap.
        noise = self.phase_variation_scheduler.sample_noise(
            size=phase.shape, enable_remap=enable_remap, col_ind=self.col_ind
        )

        phase = phase + noise

        if src == "weight":
            x = self.mrr_tr_to_weight(self.mrr_roundtrip_phase_to_tr(phase)).mul(
                S_scale[..., None]
            )  # do not modify weight inplace! we cannot call build_weight_from_phase here because it will update weight.data
        else:
            x = phase

        return x

    def _add_crosstalk_noise(self, x, src: str = "weight") -> None:
        if (not self._enable_crosstalk) or self.crosstalk_scheduler is None:
            return x  #  no noise injected

        if src == "weight":
            phase, S_scale = self.build_phase_from_weight_(
                x
            )  # we need to use this func to update S_scale and keep track of S_scale if weight gets updated.
        elif src == "phase":
            phase = x
        else:
            raise NotImplementedError

        # crosstalk_coupling_matrix = self.crosstalk_scheduler.get_crosstalk_matrix(self.phase)
        crosstalk_coupling_matrix = self.crosstalk_scheduler.get_crosstalk_matrix(phase)
        # print("coupling", crosstalk_coupling_matrix)
        phase = self.crosstalk_scheduler.apply_crosstalk(
            phase, crosstalk_coupling_matrix
        )

        if src == "weight":
            x = mzi_phase_to_out_diff(phase).mul(
                S_scale[..., None]
            )  # do not modify weight inplace! we cannot call build_weight_from_phase here because it will update weight.data
        elif src == "phase":
            x = phase
        else:
            raise NotImplementedError

        return x

    def set_light_redist(self, flag: bool = False) -> None:
        ## enable or disable light redistribution if prune_mask is available
        self._enable_light_redist = flag

    def set_input_power_gating(self, flag: bool = False, ER: float = 6) -> None:
        ## enable or disable power gating for light shutdown if prune_mask is available
        ## ER 6dB SL-MZM,
        self._enable_input_power_gating = flag
        self._input_modulator_ER = ER

    def set_output_power_gating(self, flag: bool = False) -> None:
        ## enable or disable power gating for TIA/ADC shutdown if prune_mask is available
        self._enable_output_power_gating = flag

    def _add_output_noise(self, x) -> None:
        if self.output_noise_std > 1e-6:
            if (
                self._enable_light_redist or self._enable_output_power_gating
            ) and self.prune_mask is not None:
                r, k1, k2 = self.miniblock[0], self.miniblock[-2], self.miniblock[-1]
                p, q, c = (
                    self.weight.shape[0],
                    self.weight.shape[1],
                    self.weight.shape[3],
                )  # q*c
                if self._enable_light_redist:
                    col_mask = self.prune_mask["col_mask"]  # [p,q,1,c,1,k2]
                    col_nonzeros = col_mask.sum(-1).squeeze(-1)  # [p,q,1,c]
                    factor = col_nonzeros / k2  # [p,q,1,c]
                else:
                    factor = torch.ones([p, q, 1, c], device=self.device)  # [p,q,1,c]

                if self._enable_output_power_gating:
                    row_mask = self.prune_mask["row_mask"]  # [p,q,r,1,k1,1]
                    row_mask = row_mask[..., 0, :, :].flatten(2, 3)  # [p,q,r*k1, 1]
                    factor = factor * row_mask  # [p,q,r*k1, c]
                else:
                    factor = factor.expand(-1, -1, r * k1, -1)  # [p,q,r*k1, c]

                factor = factor.permute(0, 2, 1, 3).flatten(0, 1)[
                    : x.shape[1]
                ]  # [p*r*k1, q, c] -> [out_c, q, c]

                std = factor.mul(k2**0.5).square().sum([-2, -1]).sqrt()

                std *= self.output_noise_std  # [out_c]

                noise = torch.randn_like(x)  # [bs, out_c, h, w] or [bs, out_c, q, c]

                if noise.dim() == 4:
                    noise = noise * std[..., None, None]
                elif noise.dim() == 2:
                    noise = noise * std
                else:
                    raise NotImplementedError
                x = x + noise
            else:
                vector_len = np.prod(self.weight.shape[1::2])  # q*c*k2
                noise = gen_gaussian_noise(
                    x,
                    noise_mean=0,
                    noise_std=np.sqrt(vector_len) * self.output_noise_std,
                )

                x = x + noise
        return x

    def calc_weight_MZI_power(
        self, weight=None, src: str = "weight", reduction: str = "none"
    ) -> None:

        weight = weight if weight is not None else self.weight

        if src == "weight":
            ## no inplace modification here.
            phase, _ = self.build_phase_from_weight(
                weight
            )  # we need to use this func to update S_scale and keep track of S_scale if weight gets updated.
        elif src == "phase":
            phase = weight
        else:
            raise NotImplementedError

        return self.crosstalk_scheduler.calc_MZI_power(
            phase, reduction=reduction
        )  # [p,q,r,c,k1,k2]

    def build_weight_from_phase(self, phases: Tensor) -> Tensor:
        ## inplace operation: not differentiable operation using copy_
        self.weight.data.copy_(
            mzi_phase_to_out_diff(phases).mul(self.S_scale.data[..., None])
        )
        return self.weight

    def build_phase_from_weight_(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        ## inplace operation: not differentiable operation using copy_
        phase, S_scale = self.build_phase_from_weight(weight)
        self.phase.data.copy_(phase)
        self.S_scale.data.copy_(S_scale)
        return self.phase, self.S_scale

    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        ## inplace operation: not differentiable operation using copy_
        S_scale = (
            weight.data.abs().flatten(-2, -1).max(dim=-1, keepdim=True)[0]
        )  # block-wise abs_max as scale factor

        weight = torch.where(
            S_scale[..., None] > 1e-8,
            weight.data.div(S_scale[..., None]),
            torch.zeros_like(weight.data),
        )
        phase = mzi_out_diff_to_phase(weight)
        return phase, S_scale

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        if src == "weight":
            self.build_phase_from_weight_(self.weight)
        elif src == "phase":
            self.build_weight_from_phase(self.phase)
        else:
            raise NotImplementedError

    def print_parameters(self):
        print(self.phase) if self.mode == "phase" else print(self.weight)

    def build_weight(
        self,
        weight=None,
        enable_noise: bool = True,
        enable_ste: bool = False,
    ) -> Tensor:
        if self.mode == "weight":
            weight = weight if weight is not None else self.weight
            if self.w_bit < 16:
                weight = self.weight_quantizer(weight)

            if enable_noise:
                ## use auto-diff from torch
                if enable_ste:
                    weight_tmp = weight.detach()
                else:
                    weight_tmp = weight

                phase, S_scale = self.build_phase_from_weight_(weight_tmp)
                ## step 1 add random phase variation
                phase = self._add_phase_variation(phase, src="phase")

                ## step 2 add thermal crosstalk
                phase = self._add_crosstalk_noise(phase, src="phase")

                ## reconstruct noisy weight
                weight_noisy = mzi_phase_to_out_diff(phase).mul(S_scale[..., None])

                if self._enable_output_power_gating and self.prune_mask is not None:
                    # print("Add cro")
                    # print_stat((weight_noisy - weight.data).abs())
                    weight_noisy = (
                        weight_noisy * self.prune_mask["row_mask"]
                    )  ## reapply mask to shutdown nonzero weights due to crosstalk, but gradient will still flow through the mask due to STE

                if self._enable_input_power_gating and self.prune_mask is not None:
                    ratio = 1 / 10 ** (self._input_modulator_ER / 10)
                    weight_noisy = weight_noisy * self.prune_mask[
                        "col_mask"
                    ].float().add(ratio).clamp(max=1)

                if enable_ste:
                    weight = STE.apply(
                        weight, weight_noisy
                    )  # cut off gradient for weight_noisy, only flow through weight
                else:
                    weight = weight_noisy
                self.noisy_phase = phase  # TODO: to DEBUG

        elif self.mode == "phase":
            if self.w_bit < 16:
                phase = self.phase_quantizer(self.phase)
            else:
                phase = self.phase

            if self.phase_noise_std > 1e-5:
                ### phase_S is assumed to be protected
                phase = add_gaussian_noise(
                    phase,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )

            weight = self.build_weight_from_phase(phase)
        else:
            raise NotImplementedError
        if self.weight_noise_std > 1e-6:
            weight = weight * (1 + torch.randn_like(weight) * self.weight_noise_std)
        return weight

    def forward(self, x):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""
