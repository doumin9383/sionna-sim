from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
from libs.my_configs import ResourceGridConfig, PlanarArrayConfig
from sionna.phy.nr.utils import decode_mcs_index

from sionna.phy.nr import PUSCHConfig, PUSCHReceiver, CarrierConfig, TBConfig


@dataclass
class HybridSimulationCommonConfig:
    """Common configuration for both SLS and LLS"""

    # RF/Frequency
    carrier_frequency: float = 7e9
    subcarrier_spacing: float = 60e3

    # Resource Grid Common Parameters
    rbg_size_rb: int = 6  # Definition of one Resource Block Group (Subband)

    mcs_table: int = 2
    use_transform_precoding_mcs_table: bool = False
    transform_precoding_pi2bpsk: bool = False
    direction: str = "uplink"  # "uplink", "downlink"

    # Antenna Common Parameters (Standard 3GPP Panel Layout)
    bs_polarization: str = "dual"  # "single" or "dual"
    bs_num_rows_per_panel: int = (
        2  # Number of rows per panel (real element is *2 for dual polarization)
    )
    bs_num_cols_per_panel: int = (
        2  # Number of columns per panel (real element is *2 for dual polarization)
    )
    bs_num_rows_panel: int = 4  # Number of rows per panel
    bs_num_cols_panel: int = 8  # Number of columns per panel
    bs_num_rf_chains: int = (
        bs_num_rows_panel * bs_num_cols_panel * (2 if bs_polarization == "dual" else 1)
    )  # Total Digital Ports (RF Chains) for Hybrid BF 偏波込みなら2倍

    ut_polarization: str = "dual"  # "single" or "dual"
    ut_num_rows_per_panel: int = (
        1  # Number of rows per panel (real element is *2 for dual polarization)
    )
    ut_num_cols_per_panel: int = (
        1  # Number of columns per panel (real element is *2 for dual polarization)
    )
    ut_num_rows_panel: int = 1  # Number of rows per panel
    ut_num_cols_panel: int = 2  # Number of columns per panel
    ut_num_rf_chains: int = (
        ut_num_rows_panel * ut_num_cols_panel * (2 if ut_polarization == "dual" else 1)
    )  # Total Digital Ports (RF Chains) for Hybrid BF 偏波込みなら2倍

    def __init__(self):
        self.mcs_decoder = lambda mcs: decode_mcs_index(
            mcs,
            table_index=self.mcs_table,
            is_pusch=("PUSCH" if self.direction == "uplink" else "PDSCH"),
            transform_precoding=self.use_transform_precoding_mcs_table,
            pi2bpsk=self.transform_precoding_pi2bpsk,
        )

        self.carrier_config = CarrierConfig()
        self.carrier_config.subcarrier_spacing = self.subcarrier_spacing / 1e3
        self.carrier_config.carrier_frequency = self.carrier_frequency

        self.tb_config = TBConfig()
        self.tb_config.channel_type = "PUSCH" if self.direction == "uplink" else "PDSCH"
        self.tb_config.mcs_table = self.mcs_table

    @property
    def num_bs_ant(self) -> int:
        n = (
            self.bs_num_rows_per_panel
            * self.bs_num_cols_per_panel
            * self.bs_num_panel_ports
        )
        return n * 2 if self.bs_polarization == "dual" else n

    @property
    def num_ut_ant(self) -> int:
        n = (
            self.ut_num_rows_per_panel
            * self.ut_num_cols_per_panel
            * self.ut_num_panel_ports
        )
        return n * 2 if self.ut_polarization == "dual" else n


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
    {"name": "DFT-s-OFDM", "is_dft_s": True},
    {"name": "CP-OFDM", "is_dft_s": False},
]
