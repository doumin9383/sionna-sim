from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
from libs.my_configs import ResourceGridConfig, PlanarArrayConfig


@dataclass
class HybridSimulationCommonConfig:
    """Common configuration for both SLS and LLS"""

    # RF/Frequency
    carrier_frequency: float = 7e9
    subcarrier_spacing: float = 60e3

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
