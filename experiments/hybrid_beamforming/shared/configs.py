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

    # ResourceGrid Default
    resource_grid: ResourceGridConfig = field(
        default_factory=lambda: ResourceGridConfig(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            cyclic_prefix_length=6,
            pilot_ofdm_symbol_indices=[2, 11],
        )
    )


# Common Constants / Enums for consistency across LLS/SLS
SYSTEM_MODULATIONS: Dict[str, int] = {
    "QPSK": 2,
    "16QAM": 4,
    "64QAM": 6,
    "256QAM": 8,
}
# Note: Value is bits per symbol. Sionna's mcs_index mapping is separate,
# but usually we map name -> mcs_index or name -> num_bits.
# In LLS config, it was mapping Name -> MCS Index (e.g. "QPSK": 2).
# Let's align with that usage directly or provide a mapping.
# The original LLS config used indices: {"QPSK": 2, "16QAM": 11, ...} -> These act as MCS indices.
# So we should define standard MCS indices for these modulations.

SYSTEM_MCS_INDICES: Dict[str, int] = {
    "QPSK": 2,
    "16QAM": 11,
    "64QAM": 20,
    "256QAM": 28,
}

SYSTEM_WAVEFORMS: List[Dict[str, Any]] = [
    {"name": "CP-OFDM", "is_dft_s": False},
    {"name": "DFT-s-OFDM", "is_dft_s": True},
]
