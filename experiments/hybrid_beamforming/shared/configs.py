from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
from libs.my_configs import ResourceGridConfig, PlanarArrayConfig


@dataclass
class HybridSimulationCommonConfig:
    """Common configuration for both SLS and LLS"""

    # RF/Frequency
    carrier_frequency: float = 7e9
    subcarrier_spacing: float = 60e3

    # Resource Grid Common Parameters
    rbg_size_rb: int = 6  # Definition of one Resource Block Group (Subband)

    # Antenna Common Parameters (Standard 3GPP Panel Layout)
    bs_num_rows: int = 8
    bs_num_cols: int = 16
    bs_polarization: str = "dual"  # "single" or "dual"

    ut_num_rows: int = 2
    ut_num_cols: int = 1
    ut_polarization: str = "dual"

    @property
    def num_bs_ant(self) -> int:
        n = self.bs_num_rows * self.bs_num_cols
        return n * 2 if self.bs_polarization == "dual" else n

    @property
    def num_ut_ant(self) -> int:
        n = self.ut_num_rows * self.ut_num_cols
        return n * 2 if self.ut_polarization == "dual" else n

    # Resource Grid Default
    # 共通のグリッド設定があればここに
    # resource_grid: ResourceGridConfig = field(
    #     default_factory=lambda: ResourceGridConfig(
    #         num_ofdm_symbols=14,
    #         fft_size=64,
    #         subcarrier_spacing=30e3,
    #         cyclic_prefix_length=6,
    #     )
    # )
