# basic simulator for photonic crossbar
import math
from .photonic_core_base import PhotonicCore
import torch
# from core.models.layers.utils import MZIPowerEvaluator

#  area: um^2, prec: bit, power: mw, sample_rate: GSample/s
DAC_list = {
    1: {"area": 11000, "prec": 12, "power": 169, "sample_rate": 14, "FoM": 33.6},
    2: {"area": 11000, "prec": 8, "power": 50, "sample_rate": 14, "FoM": None},
    3: {"area": 500000, "prec": 8, "power": 20, "sample_rate": 5, "FoM": None},
    4: {"area": 500000, "prec": 8, "power": 20, "sample_rate": 0.001, "FoM": None},
    5: {"area": 500000, "prec": 8, "power": 20, "sample_rate": 0.001, "FoM": None},
}

# 1:  A 10GS/s 8b 25fJ/c-s 2850um2 Two-Step Time-Domain ADC Using Delay-Tracking Pipelined-SAR TDC with 500fs Time Step in 14nm CMOS Technology
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9731625
ADC_list = {
    1: {"area": 2850, "prec": 8, "power": 14.8, "sample_rate": 10, "type": "sar"},
    2: {"area": 100000, "prec": 8, "power": 7.5, "sample_rate": 5, "type": "sar"}
}

__all__ = ["PhotonicCrossbar"]


class PhotonicCrossbar(PhotonicCore):
    """Simulation of photonic crossbar arch
    Start with a laser, then multi-wavelength (wavelength num: D) comb, then high-speed modulator for different wavelength.
    At each cross-point, coupler will couple 1/N (N = max(Nw, Nh)) portion of light into individual len-D vector product following one PS + one coupler.
    Use differential photodetector to do detection.
    """

    def __init__(
            self,
            core_width=32,
            core_height=32,
            num_wavelength=32,
            in_bit=4,
            w_bit=4,
            act_bit=4,
            switch_bit = 8,
            config=None,
    ) -> None:
        super().__init__()
        # basic photonic crossbar params: Cw, Ch, Nw
        self.core_width = core_width  # core_width of photonic crossbar

        self.core_height = core_height  # core_height of photonic crossbar
        print(f"This is core height:{self.core_height}")
        self.num_pe_per_tile = config.arch.num_pe_per_tile
        self.num_tiles = config.arch.num_tiles
        self.sharing_factor_r = config.arch.r
        self.sharing_factor_c = config.arch.c

        # num of wavelength used in WDM for len-k vec computation
        self.num_wavelength = num_wavelength
        self.interval_s = config.device.phase_shifter_interval.interv_s
        self.interval_v = config.device.phase_shifter_interval.interv_v
        self.interval_h = config.device.phase_shifter_interval.interv_h
        self.structural_sparsity = config.arch.structural_sparsity
        self.tunable_splitter = config.arch.tunable_splitter

        # precision params
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.act_bit = act_bit
        self.switch_bit = switch_bit
        self.oedac_partition = config.arch.oedac_partition
        # print(self.in_bit)

        # DAC and ADC params
        self.core_DAC_power = 0
        self.core_DAC_prec = 4
        self.core_DAC_area = 0
        self.core_DAC_sampling_rate = 1
        self.core_ADC_power = 0
        self.core_ADC_prec = 4
        self.core_ADC_area = 0
        self.core_ADC_sampling_rate = 1


        self.__obtain_ADC_param(config.core.interface.ADC)
        self.__obtain_DAC_param(config.core.interface.DAC)
        self.__obtain_DAC_weight_param(config.core.interface.DAC_weight)
        self.__obtain_DAC_switch_param(config.core.interface.DAC_switch)
        self.__obtain_TIA_param(config.core.interface.TIA)
        self.__obtain_sparsity_param(config.core.sparsity)

        self.__obtain_laser_param(config.device.laser)
        self.__obtain_extinction_rate(config.device.mzi_modulator)
        self.__obtain_modulator_param(config.device.mzi_modulator)
        self.__obtain_mrr_router_param(config.device.mrr_router)
        self.__obtain_phase_shifter_param(config.device.phase_shifter)
        self.__obtain_direction_coupler_param(config.device.direction_coupler)
        self.__obtain_integrator_param(config.device.integrator)
        self.__obtain_photo_detector_param(config.device.photo_detector)
        self.__obtain_y_branch_param(config.device.y_branch)
        self.__obtain_mmi_param(getattr(config.device, "mmi", None))
        self.__obtain_crossing_param(getattr(config.device, "crossing", None))
        self.__obtain_waveguide_param(getattr(config.device, "waveguide", None))
        self.__obtain_micro_comb_param(config.device.micro_comb)
        self.__obtain_sharing_factors()

        # set work freq
        self.work_freq = config.core.work_freq if config is not None else 1  # GHz
        self.laser_power_scale = config.core.laser_power_scale if config is not None else 1  # scaling down factor for laser power due to time accumulation.

        # cal params
        # first obtain insertion loss
        self.insertion_loss = None
        self.core_power = None
        self.insertion_loss_computation = None
        self.insertion_loss_modulation = None
        self.cal_insertion_loss(print_msg=True)
        self.cal_laser_power(print_msg=True)
        self.cal_modulator_param(print_msg=True)
        self.cal_ADC_param(print_msg=True)
        self.cal_DAC_param(print_msg=True)
        self.cal_DAC_weight_param(print_msg=True)
        self.cal_DAC_switch_param(print_msg=True)
        self.cal_core_power()
        self.cal_architecture_power()
        self.cal_node_area()

    def set_precison(self, in_bit, w_bit, act_bit):
        # set input, weight and activation bit-core_width to scale AD/DA energy consumption.
        if (
                (in_bit != self.in_bit)
                or (w_bit != self.w_bit)
                or (act_bit != self.act_bit)
        ):
            print(
                f"core precision change from in-{self.in_bit} w-{self.w_bit} act-{self.act_bit} to in-{in_bit} w-{w_bit} act-{act_bit}"
            )
            self.in_bit = in_bit
            self.w_bit = w_bit
            self.a_bit = act_bit
            self.cal_ADC_param()
            self.cal_DAC_param()
            self.cal_modulator_param()

    def set_work_frequency(self, work_freq):
        # set work frequency -> energy should be power * frequency
        if self.work_freq != work_freq:
            # recalculate fre-realted params
            # modulator dynamic power, adc and dac power
            print(
                f"Work frequency of the photonic core change from {self.work_freq} GHz to {work_freq} GHz"
            )
            self.work_freq = work_freq
            self.cal_ADC_param()
            self.cal_DAC_param()
            self.cal_modulator_param()

    def cal_laser_power(self, print_msg=False):
        if (
                self.insertion_loss_modulation is None
                or self.insertion_loss_computation is None
        ):
            self.cal_insertion_loss()
        # method 1: follow DAC crosslight to compute laser power
        # P_laser - S_detector >= P_photoloss (dbm)
        # the laser will be split into how many parts: num_wavelength, then (N + M)
        # self.num_wavelength: all wave summed up -> output energy > sensitivity
        IL = self.insertion_loss

        P_laser_dbm = self.photo_detector_sensitivity + IL  # * self.core_height
        self.laser_power = (
                10 ** (P_laser_dbm / 10) / self.laser_wall_plug_eff * 2 ** self.act_bit * self.sharing_factor_r
        )

        ## extra power due to min_rx_power requirement for PD to gaurantee a good enough eye digram, i.e., SNR
        self.laser_power += (
                10 ** ((self.photo_detector_min_rx_power + IL) / 10)
                / self.laser_wall_plug_eff
        )
        # print(f"Multiply with {10 ** ((self.photo_detector_sensitivity + self.insertion_loss_modulation + self.insertion_loss_computation) / 10)}")
        # photon_energy = 1.28e-19 # joule at 1.55um
        photon_energy = 6.626e-34 * 2.99e8 / (self.laser_wavelength * 1e-6)  # joule

        laser_power2 = (
                self.core_height
                * self.core_width
                / 0.2
                * (2 ** (2 * self.act_bit + 1))
                * photon_energy
                * self.work_freq
                * 1e9
                * 1e3
        )
        print(laser_power2)

        if print_msg:
            print(
                f"required laser power is {self.laser_power} mW (electrical) and {self.laser_power * self.laser_wall_plug_eff} mW (optical) with {P_laser_dbm} db requirement"
            )

        if self.extinction_rate is not None:
            scale_factor = 1 / (1 - 0.1 ** (self.extinction_rate / 10))
            self.laser_power *= scale_factor
            if print_msg:
                print(
                    f"Need to increase the laser power by {scale_factor} to maintain the calculation accuracy"
                )

        ## reduce laser power by T factor if time accumulation is T times.
        if self.laser_power_scale is not None:
            self.laser_power /= (self.laser_power_scale * self.num_pe_per_tile)
            if print_msg:
                print(
                    f"required laser power (scaled by {self.laser_power_scale}) is {self.laser_power} mW (electrical) and {self.laser_power * self.laser_wall_plug_eff} mW (optical) with {P_laser_dbm} db requirement"
                )

    def cal_insertion_loss(self, print_msg=False):
        "Function to compute insertion loss"
        # ignor grating coupler since we assume this is a on-chip laser

        self.insertion_loss_modulation = (
                self.modulator_insertion_loss
                + self.mrr_router_insertion_loss * (2 if self.num_wavelength > 1 else 0)
        )
        if not self.mmi_available:
            self.insertion_loss_modulation += self.y_branch_insertion_loss * math.ceil(
                math.log2(max(self.core_height, self.core_width)))
        else:
            self.insertion_loss_modulation += self.mmi_insertion_loss

        # 1 splitter + 1 ps + 1 dc
        #
        self.insertion_loss_computation = (
                self.y_branch_insertion_loss
                + self.phase_shifter_insertion_loss
                + self.direction_coupler_insertion_loss
        )
        self.splitter_insertion_loss = (math.log2(self.core_width) * self.insertion_loss_computation) if self.tunable_splitter else (math.log2(self.core_width) * self.y_branch_insertion_loss)
        self.insertion_loss = (
                self.insertion_loss_computation
                + self.insertion_loss_modulation
                + 10 * math.log10(self.core_width * self.core_height)
                + self.laser_coupling_loss
                + self.splitter_insertion_loss
        )

    def __obtain_y_branch_param(self, config=None):
        if config is not None:
            self.y_branch_length = config.length
            self.y_branch_width = config.width
            self.y_branch_insertion_loss = config.insertion_loss
        else:
            self.y_branch_length = 75
            self.y_branch_width = 3.9
            self.y_branch_insertion_loss = 0.1

    def __obtain_switch_param(self, config=None):
        if config is not None:
            self.switch_length = self.node_length
            self.switch_width = self.node_width
            self.switch_interval_s = config.switch_interv_s
            self.switch_interval_h = config.switch_interv_h
        else:
            self.switch_length = 0
            self.switch_width = 0
            self.switch_interval_s = 0
            self.switch_interval_h = 0

    def __obtain_sharing_factors(self):
        print(self.num_pe_per_tile)
        print(self.num_tiles)
        assert (self.num_pe_per_tile % self.sharing_factor_c) == 0, "cannot obatin sharing factor due to not diviable sharing factor"
        assert (self.num_tiles % self.sharing_factor_r) == 0, "cannot obatin sharing factor due to not diviable sharing factor"
        self.sharing_c = self.num_pe_per_tile / self.sharing_factor_c
        self.sharing_r = self.num_tiles / self.sharing_factor_r

    def __obtain_sparsity_param(self, config=None):
        if config is not None:
            self.empty_rows = config.rows
            self.empty_column = config.column
        else:
            self.empty_rows = 0
            self.empty_column = 0

    def __obtain_mmi_param(self, config=None):
        if config is not None:
            if self.core_height == 10:
                self.mmi_length = config["1x10"].length
                self.mmi_width = config["1x10"].width
                self.mmi_insertion_loss = config["1x10"].insertion_loss
            else:
                factor = self.core_height / 10
                print(f"This is factor:{factor}")
                self.mmi_length = config["1x10"].length * factor
                self.mmi_width = config["1x10"].width * factor
                self.mmi_insertion_loss = config["1x10"].insertion_loss
            self.mmi_available = True
        else:
            self.mmi_available = False
            print("False")

    def __obtain_crossing_param(self, config=None):
        self.crossing_length = config.length
        self.crossing_width = config.width
        self.crossing_insertion_loss = config.insertion_loss

    def __obtain_waveguide_param(self, config=None):
        self.waveguide_length = config.length
        self.waveguide_insertion_loss = config.insertion_loss

    def __obtain_micro_comb_param(self, config=None):
        if config is not None:
            self.micro_comb_length = config.length
            self.micro_comb_width = config.width
        else:
            self.micro_comb_length = 1184
            self.micro_comb_width = 1184
        self.micro_comb_area = self.micro_comb_length * self.micro_comb_width

    def __obtain_photo_detector_param(self, config=None):
        if config is not None:
            self.photo_detector_power = config.power
            self.photo_detector_length = config.length
            self.photo_detector_width = config.width
            self.photo_detector_sensitivity = config.sensitivity
            self.photo_detector_min_rx_power = config.min_rx_power
        else:
            self.photo_detector_power = 2.8
            self.photo_detector_length = 40
            self.photo_detector_width = 40
            self.photo_detector_sensitivity = -25
            self.photo_detector_min_rx_power = -10  # dbm

    def __obtain_direction_coupler_param(self, config=None):
        if config is not None:
            self.direction_coupler_length = config.length
            self.direction_coupler_width = config.width
            self.direction_coupler_insertion_loss = config.insertion_loss
        else:
            self.direction_coupler_length = 75
            self.direction_coupler_width = 10
            self.direction_coupler_insertion_loss = 0.3

    def __obtain_phase_shifter_param(self, config=None):
        if config is not None:
            self.phase_shifter_power_dynamic = config.dynamic_power
            self.phase_shifter_power_static = config.static_power
            self.phase_shifter_length = config.length
            self.phase_shifter_width = config.width
            self.phase_shifter_insertion_loss = config.insertion_loss
        else:
            self.phase_shifter_power_dynamic = 0
            self.phase_shifter_power_static = 0
            self.phase_shifter_length = 200
            self.phase_shifter_width = 34
            self.phase_shifter_insertion_loss = 0.2

    def __obtain_mrr_router_param(self, config=None):
        if config is not None:
            self.mrr_router_power = config.static_power
            self.mrr_router_length = config.length
            self.mrr_router_width = config.width
            self.mrr_router_insertion_loss = config.insertion_loss
        else:
            self.mrr_router_power = 2.4
            self.mrr_router_length = 20
            self.mrr_router_width = 20
            self.mrr_router_insertion_loss = 0.25

    def __obtain_modulator_param(self, config=None):
        if config is not None:
            self.modulator_type = config.type
            assert self.modulator_type == "mzi"
            self.modulator_energy_per_bit = config.energy_per_bit
            self.modulator_power_static = config.static_power
            self.modulator_length = config.length
            self.modulator_width = config.width
            self.modulator_insertion_loss = config.insertion_loss
        else:
            self.modulator_energy_per_bit = 400
            self.modulator_static_power = 0
            self.modulator_length = 300
            self.modulator_width = 50
            self.modulator_insertion_loss = 0.8

    def __obtain_integrator_param(self, config=None):
        if config is not None:
            self.integrator_static_power = config.static_power
            self.integrator_area = config.area
            self.integrator_insertion_loss = config.insertion_loss
        else:
            self.integrator_static_power = 348
            self.integrator_area = 1500000
            self.integrator_insertion_loss = 10

    def cal_modulator_param(self, print_msg=False):
        # indepent to bit width
        self.modulator_power_dynamic = (
                self.modulator_energy_per_bit * self.work_freq * 1e-3
        )  # mW

    def __obtain_extinction_rate(self, config=None):
        if config is not None:
            self.extinction_rate = config.extinction_rate
        else:
            self.extinction_rate = 0

    def __obtain_laser_param(self, config=None):
        if config is not None:
            self.laser_power = config.power
            self.laser_length = config.length
            self.laser_width = config.width
            self.laser_area = self.laser_length * self.laser_width
            self.laser_wall_plug_eff = config.wall_plug_eff
            self.laser_coupling_loss = config.coupling_loss
            self.laser_wavelength = config.wavelength
        else:
            self.laser_power = 0.5
            self.laser_length = 400
            self.laser_width = 300
            self.laser_area = self.laser_length * self.laser_width
            self.laser_wall_plug_eff = 0.25
            self.laser_coupling_loss = 2  # dB
            self.laser_wavelength = 1.55  # um

    def __obtain_TIA_param(self, config=None):
        if config is not None:
            self.TIA_power = config.power
            self.TIA_area = config.area
        else:
            self.TIA_power = 3
            self.TIA_area = 5200

    def __obtain_DAC_param(self, config=None):
        if config is not None:
            DAC_choice = config.choice
            assert DAC_choice in [1, 2, 3]
            self.chosen_DAC_list = DAC_list[DAC_choice]
            self.DAC_area = self.chosen_DAC_list["area"]
            self.DAC_prec = self.chosen_DAC_list["prec"]
            self.DAC_power = self.chosen_DAC_list["power"]
            self.DAC_sample_rate = self.chosen_DAC_list["sample_rate"]
            self.DAC_FoM = self.chosen_DAC_list["FoM"]
        else:
            raise NotImplementedError

    def cal_DAC_param(self, print_msg=False):
        # convert power to desired freq and bit width
        assert (
                self.in_bit <= self.DAC_prec
        ), f"Got input bit {self.in_bit} exceeds the DAC precision limit"
        in_bit = self.in_bit / self.oedac_partition
        if self.DAC_FoM is not None:
            # following 2 * FoM * nb * Fs / Br (assuming Fs=Br)
            self.core_DAC_power = 2 * self.DAC_FoM * in_bit * self.work_freq * 1e-3
        else:
            # P \propto 2**N/(N+1) * f_clk
            self.core_DAC_power = (
                    self.DAC_power
                    * (2 ** in_bit / (in_bit + 1))
                    / (2 ** self.DAC_prec / (self.DAC_prec + 1))
                    * self.work_freq
                    / self.DAC_sample_rate
            )
        # TODO(hqzhu): add tech node scaling for area here
        self.core_DAC_power *= self.oedac_partition
        self.core_DAC_area = self.DAC_area * self.oedac_partition
        self.core_DAC_prec = self.in_bit
        self.core_DAC_sampling_rate = self.work_freq
        if print_msg:
            print(
                f"The {self.core_DAC_prec}-bit crossbar DAC power @{self.core_DAC_sampling_rate}GHz is {self.core_DAC_power} mW"
            )
            print(f"--crossbar DAC area is {self.core_DAC_area} um^2")

    def __obtain_DAC_weight_param(self, config=None):
        if config is not None:
            DAC_choice = config.choice
            assert DAC_choice in [1, 2, 3, 4]
            self.chosen_DAC_weight_list = DAC_list[DAC_choice]
            self.DAC_weight_area = self.chosen_DAC_list["area"]
            self.DAC_weight_prec = self.chosen_DAC_list["prec"]
            self.DAC_weight_power = self.chosen_DAC_list["power"]
            self.DAC_weight_sample_rate = self.chosen_DAC_list["sample_rate"]
            self.DAC_weight_FoM = self.chosen_DAC_list["FoM"]
        else:
            raise NotImplementedError

    def cal_DAC_weight_param(self, print_msg=False):
        # convert power to desired freq and bit width
        assert (
                self.w_bit <= self.DAC_weight_prec
        ), f"Got input bit {self.w_bit} exceeds the DAC precision limit"
        if self.DAC_weight_FoM is not None:
            # following 2 * FoM * nb * Fs / Br (assuming Fs=Br)
            self.core_DAC_weight_power = 2 * self.DAC_weight_FoM * self.w_bit * self.work_freq * 1e-3
        else:
            # P \propto 2**N/(N+1) * f_clk
            self.core_DAC_weight_power = (
                    self.DAC_weight_power
                    * (2 ** self.w_bit / (self.w_bit + 1))
                    / (2 ** self.DAC_weight_prec / (self.DAC_weight_prec + 1))
                    * 0.001
                    / self.DAC_weight_sample_rate
            )

        # TODO(hqzhu): add tech node scaling for area here
        self.core_DAC_weight_area = self.DAC_weight_area
        self.core_DAC_weight_prec = self.in_bit
        self.core_DAC_weight_sampling_rate = 0.001
        if print_msg:
            print(
                f"The {self.core_DAC_weight_prec}-bit crossbar DAC power @{self.core_DAC_weight_sampling_rate}GHz is {self.core_DAC_weight_power} mW"
            )
            print(f"--crossbar DAC area is {self.core_DAC_weight_area} um^2")

    def __obtain_DAC_switch_param(self, config=None):
        if config is not None:
            DAC_choice = config.choice
            assert DAC_choice in [1, 2, 3, 4]
            self.chosen_DAC_switch_list = DAC_list[DAC_choice]
            self.DAC_switch_area = self.chosen_DAC_list["area"]
            self.DAC_switch_prec = self.chosen_DAC_list["prec"]
            self.DAC_switch_power = self.chosen_DAC_list["power"]
            self.DAC_switch_sample_rate = self.chosen_DAC_list["sample_rate"]
            self.DAC_switch_FoM = self.chosen_DAC_list["FoM"]
        else:
            raise NotImplementedError

    def cal_DAC_switch_param(self, print_msg=False):
        if self.DAC_switch_FoM is not None:
            # following 2 * FoM * nb * Fs / Br (assuming Fs=Br)
            self.core_DAC_switch_power = 2 * self.DAC_switch_FoM * 1 * self.work_freq * 1e-3
        else:
            # P \propto 2**N/(N+1) * f_clk
            self.core_DAC_switch_power = (
                    self.DAC_switch_power
                    * (2 ** self.switch_bit  / (self.switch_bit  + 1))
                    / (2 ** self.DAC_switch_prec / (self.DAC_switch_prec + 1))
                    * 0.001
                    / self.DAC_switch_sample_rate
            )

        # TODO(hqzhu): add tech node scaling for area here
        self.core_DAC_switch_area = self.DAC_switch_area
        self.core_DAC_switch_prec = self.in_bit
        self.core_DAC_switch_sampling_rate = 0.001
        if print_msg:
            print(
                f"The {self.core_DAC_switch_prec}-bit crossbar DAC power @{self.core_DAC_switch_sampling_rate}GHz is {self.core_DAC_switch_power} mW"
            )
            print(f"--crossbar DAC area is {self.core_DAC_switch_area} um^2")

    def __obtain_ADC_param(self, config=None):
        if config is not None:
            ADC_choice = config.choice
            self.core_ADC_sharing_factor = config.sharing_factor
            assert ADC_choice in [1, 2]
            self.chosen_ADC_list = ADC_list[ADC_choice]
            self.ADC_area = self.chosen_ADC_list["area"]
            self.ADC_prec = self.chosen_ADC_list["prec"]
            self.ADC_power = self.chosen_ADC_list["power"]
            self.ADC_sample_rate = self.chosen_ADC_list["sample_rate"]
            self.ADC_type = self.chosen_ADC_list["type"]
        else:
            raise NotImplementedError

    def cal_ADC_param(self, print_msg=False):
        assert (
                self.act_bit <= self.ADC_prec
        ), f"Got input bit {self.act_bit} exceeds the ADC precision limit"
        # convert power to desired freq and bit width
        if self.ADC_type == "sar":
            # P \propto N
            self.core_ADC_power = (
                    self.ADC_power
                    * self.work_freq
                    / self.ADC_sample_rate
                    * (self.act_bit / self.ADC_prec)
            )
        elif self.ADC_type == "flash":
            # P \propto (2**N - 1)
            self.core_ADC_power = (
                    self.ADC_power
                    * self.work_freq
                    / self.ADC_sample_rate
                    * ((2 ** self.act_bit - 1) / (2 ** self.ADC_prec - 1))
            )

        # TODO(hqzhu): add tech node scaling for area here
        self.core_ADC_area = self.ADC_area
        self.core_ADC_prec = self.act_bit
        self.core_ADC_sampling_rate = self.work_freq
        if print_msg:
            print(
                f"The {self.core_ADC_prec}-bit crossbar ADC power @{self.core_ADC_sampling_rate}GHz is {self.core_ADC_power} mW"
            )
            print(f"--crossbar ADC area is {self.core_ADC_area} um^2")

    def calc_total_energy(self, cycle_dict, dst_scheduler, model, IG_flag, OG_flag):
        work_freq = self.work_freq
        layer_energy= {}
        layer_energy_breakdown = {}
        for name, m in model.named_modules():
            if isinstance(m, model._conv_linear):  # no last fc layer
                energy_breakdown = {}
                p, q, r, c, k1, k2 = m.weight.shape
                mask = m.prune_mask
                # First get switch energy
                if mask is not None:
                    switch_power = dst_scheduler.cal_ports_power(mask["col_mask"].flatten(0, -2)).sum().item()
                    RC_empty_rows = ((~mask["row_mask"]).sum((-2, -1)).expand(-1, -1, -1, self.sharing_factor_c).permute(0, 2, 1, 3).flatten())
                    RC_empty_cols = ((~mask["col_mask"]).sum((-2, -1)).expand(-1, -1, self.sharing_factor_r, -1).permute(0, 2, 1, 3).flatten())
                    total_empty_elemetns = ((~mask.data).sum((-2, -1)).permute(0, 2, 1, 3).flatten())

                    # print(RC_empty_rows, RC_empty_cols, total_empty_elemetns)
                else:
                    switch_power = 0
                    RC_empty_rows = RC_empty_cols = total_empty_elemetns = [torch.zeros(1)]*(p*q*r*c)

                # Get all energy besides weight MZI
                input_power_dac_total = input_power_modulation_total = core_photo_detector_power_total = core_TIA_power_total = core_power_adc_total = 0

                for i in range(p*q*r*c):
                    input_power_dac, input_power_modulation, core_photo_detector_power, core_TIA_power, core_power_adc = self.calc_core_power(RC_empty_rows[i].item(), RC_empty_cols[i].item(), total_empty_elemetns[i].item(), IG_flag, OG_flag)
                    input_power_dac_total += input_power_dac
                    input_power_modulation_total += input_power_modulation
                    core_photo_detector_power_total += core_photo_detector_power
                    core_TIA_power_total += core_TIA_power
                    core_power_adc_total += core_power_adc

                architecture_wise_power = (self.calc_architecture_power(energy_breakdown, input_power_dac_total, input_power_modulation_total, core_photo_detector_power_total, core_TIA_power_total, core_power_adc_total)) + switch_power
                energy_breakdown["switch"] = switch_power
                energy_breakdown = {key: (value * cycle_dict[name][0] / work_freq / 1e9) for key, value in energy_breakdown.items()}
                # print(cycle_dict)
                architecture_wise_energy = architecture_wise_power * cycle_dict[name][0] / work_freq / 1e9
                layer_energy[name] = architecture_wise_energy
                layer_energy_breakdown[name] = energy_breakdown

                newtwork_energy_breakdown = {}

                # Iterate through each layer's dictionary
                for layer, components in layer_energy_breakdown.items():
                    for component, value in components.items():
                        if component in newtwork_energy_breakdown:
                            newtwork_energy_breakdown[component] += value
                        else:
                            newtwork_energy_breakdown[component] = value

                total_energy = sum(layer_energy.values())

        return layer_energy, layer_energy_breakdown, newtwork_energy_breakdown, total_energy

    def calc_core_power(
            self,
            empty_rows: int = 0,
            empty_cols: int = 0,
            total_empty_elements: int = 0,
            IG_flag: bool = 0,
            OG_flag: bool = 0,
            ):
        
        # self.input_power_laser = self.laser_power
        if IG_flag:
            self.input_power_dac = (self.core_width - empty_cols) * self.num_wavelength * self.core_DAC_power
            self.input_power_modulation = (self.core_width - empty_cols) * self.num_wavelength \
                                      * (self.modulator_power_static + self.modulator_power_dynamic
                                         + self.mrr_router_power * (2 if self.num_wavelength > 1 else 0))
        else:
            self.input_power_dac = self.core_width* self.num_wavelength * self.core_DAC_power
            self.input_power_modulation = self.core_width * self.num_wavelength \
                                      * (self.modulator_power_static + self.modulator_power_dynamic
                                         + self.mrr_router_power * (2 if self.num_wavelength > 1 else 0))


        if OG_flag:
            self.core_TIA_power = (
                    (self.core_height - empty_rows)
                    * self.TIA_power
            )

            self.core_power_adc = (
                    (self.core_height - empty_rows)
                    * self.core_ADC_power
            )
        else:
            self.core_TIA_power = (
                    (self.core_height)
                    * self.TIA_power
            )

            self.core_power_adc = (
                    (self.core_height)
                    * self.core_ADC_power
            )            


        self.core_photo_detector_power = (
                ((self.core_height * self.core_width) - total_empty_elements)
                * self.photo_detector_power
                * 2
        )



        return self.input_power_dac, self.input_power_modulation, self.core_photo_detector_power, self.core_TIA_power, self.core_power_adc

    def calc_architecture_power(self, layer_energy_breakdown, input_power_dac, input_power_modulation, core_photo_detector_power, core_TIA_power, core_power_adc):

        self.architecture_photo_detector_power = core_photo_detector_power

        self.architecture_modulation_power = input_power_modulation / self.sharing_factor_r

        self.architecture_dac_power = input_power_dac / self.sharing_factor_r

        self.architecture_TIA_power =  core_TIA_power / self.sharing_factor_c

        self.architecture_adc_power = core_power_adc / self.sharing_factor_c

        layer_energy_breakdown["PD"] = self.architecture_photo_detector_power
        layer_energy_breakdown["MZM"] = self.architecture_modulation_power
        layer_energy_breakdown["HDAC"] = self.architecture_dac_power
        layer_energy_breakdown["TIA"] = self.architecture_TIA_power
        layer_energy_breakdown["ADC"] = self.architecture_adc_power 

        self.architecture_power = (
                self.architecture_modulation_power
                + self.architecture_dac_power
                + self.architecture_photo_detector_power
                + self.architecture_TIA_power
                + self.architecture_adc_power
        )

        # print(f"This is common core power in architecture level:{self.architecture_power}")

        architecture_power = self.architecture_power
        return architecture_power

    def cal_architecture_area(self):

        self.architecture_node_area = (self.num_tiles * self.num_pe_per_tile) * self.total_node_area

        self.architecture_dac_weight_area = (self.num_tiles * self.num_pe_per_tile) * self.dac_weight_area

        self.architecture_photo_detector_area = (self.num_tiles * self.num_pe_per_tile) * self.single_core_photodetector_area


        self.architecture_mmi_M_area = (self.num_tiles * self.num_pe_per_tile) * self.mmi_M_area

        self.architecture_dac_switch_area = (self.sharing_r * self.num_pe_per_tile) * self.single_switch_DAC_area

        self.architecture_dac_area = (self.sharing_r * self.num_pe_per_tile) * self.dac_area

        self.architecture_mzi_modulator_area = (self.sharing_r * self.num_pe_per_tile) * self.mzi_modulator_area

        self.architecture_switch_splitter_area = (self.sharing_r * self.num_pe_per_tile) * self.single_cascaded_splitter_area


        self.architecture_ADC_area = (self.num_tiles * self.sharing_c) * self.single_core_ADC_area

        self.architecture_TIA_area = (self.num_tiles * self.sharing_c) * self.single_core_TIA_area


        self.architecture_area = (
                self.architecture_dac_area
                + self.architecture_photo_detector_area
                + self.architecture_mmi_M_area
                + self.architecture_node_area
                + self.architecture_mzi_modulator_area
                + self.architecture_switch_splitter_area
                + self.architecture_TIA_area
                + self.architecture_ADC_area
        )

        print(f"This is common core area in architecture level:{self.architecture_area}")

        architecture_area = self.architecture_area

        return architecture_area

    def calc_single_splitter_width(self):
        self.switch_width = self.phase_shifter_width + self.switch_interval_s

    def cal_core_area(self):

        self.dac_area = self.core_width * self.core_DAC_area * self.num_wavelength
        self.mzi_modulator_area = self.core_width * self.modulator_width * self.modulator_length * self.num_wavelength

        self.dac_weight_area = self.core_height * self.core_width * self.core_DAC_weight_area
        self.single_core_photodetector_area = (self.core_width * self.core_height) * (
                    2 * self.photo_detector_width * self.photo_detector_length)

        self.total_node_area = ((self.core_width - 1) * self.interval_v + self.node_length) * (
                    (self.core_height - 1) * self.interval_h + self.node_width)



        self.mmi_M_area = self.core_width * self.mmi_width * self.mmi_length
        self.single_switch_DAC_area = (self.core_width - 1) * self.DAC_switch_area

        self.calc_single_splitter_width()
        self.single_cascaded_splitter_area = (self.node_length * ((self.core_width - 2) * self.switch_interval_h + self.switch_width))



        self.single_core_TIA_area = self.core_height * self.TIA_area
        self.single_core_ADC_area = self.core_height * self.core_ADC_area

        core_area = 0
        common_core_area = 0

        return core_area, common_core_area

    def cal_node_area(self):
        self.node_length = (
            # max(self.phase_shifter_length, (self.direction_coupler_width + self.photo_detector_length))
            # + self.y_branch_length
            # + (3*5)
            # + 10
                self.y_branch_length
                + self.phase_shifter_length
                + self.direction_coupler_length
                + 23
        )

        self.node_width = (

            self.phase_shifter_width + self.interval_s
        )


        self.node_area = self.node_width * self.node_length

        node_area = self.node_area

        return node_area


    def cal_D2A_energy(self):
        self.D2A_energy = self.core_DAC_power / (self.work_freq)  # pJ
        return self.D2A_energy

    def cal_integrator_energy(self):
        self.integrator_energy = self.integrator_static_power / (self.work_freq)
        return self.integrator_energy

    def cal_TX_energy(self):
        # mzi modulator
        self.TX_energy = (
                             (
                                     self.modulator_power_dynamic
                                     + self.modulator_power_static
                                     + self.mrr_router_power * (2 if self.num_wavelength > 1 else 0)
                             )
                         ) / self.work_freq
        return self.TX_energy

    def cal_A2D_energy(self):
        self.A2D_energy = self.core_ADC_power / (self.work_freq)  # pJ
        return self.A2D_energy

    def cal_RX_energy(self):
        # crossbar use two detector
        self.RX_energy = (
                self.photo_detector_power / (self.work_freq) * 2
                + self.TIA_power / self.work_freq
        )
        self.photo_detector_energy = self.photo_detector_power / (self.work_freq) * 2
        self.TIA_energy = self.TIA_power / self.work_freq
        return self.RX_energy

    def cal_comp_energy(self):
        self.comp_energy = (
                                   self.phase_shifter_power_dynamic + self.phase_shifter_power_static
                           ) / (self.work_freq)
        return self.comp_energy

    def cal_laser_energy(self):
        self.laser_energy = self.laser_power / (self.work_freq)
        return self.laser_energy