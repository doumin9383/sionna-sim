from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
import numpy as np

# 基底クラスのインポート
from libs.my_configs import (
    ResourceGridConfig,
    PlanarArrayConfig,
)
from experiments.hybrid_beamforming.shared.configs import HybridSimulationCommonConfig

from sionna.phy.channel.tr38901 import PanelArray


@dataclass
class HybridSLSConfig(HybridSimulationCommonConfig):
    """System Level Simulation Configuration for experiments/hybrid_beamforming/sls"""

    # Simulation Control
    batch_size: int = 1  # Debug: 1
    num_rings: int = 1
    topology_type: str = "HexGrid"  # "HexGrid", "Custom", etc.
    num_ut_per_sector: int = 1  # Production: 4

    @property
    def num_bs(self):
        if self.topology_type == "HexGrid":
            # Lazy import to avoid circular dependency or just standard import earlier
            from sionna.sys import get_num_hex_in_grid

            num_cells = get_num_hex_in_grid(self.num_rings)
            return num_cells * 3
        else:
            raise NotImplementedError(
                f"Topology type {self.topology_type} not supported for auto num_bs calc."
            )

    num_ut_drops: int = 2  # Debug: 2. Number of random topology drops.
    num_slots: int = 1  # Fixed to 1 for snapshot simulation
    precoding_granularity: str = "Wideband"  # "Narrowband", "Subband", "Wideband"
    use_rbg_granularity: bool = True  # If True, calculate channel only at RBG centers
    num_neighbors: int = (
        16  # For spatial masking: num BS per UT to calculate channel for
    )

    num_rb = 66
    num_subcarriers = num_rb * 12

    # RF/Frequency (Inherited from HybridSimulationCommonConfig)
    # carrier_frequency: float = 3.5e9
    # subcarrier_spacing: float = 30e3

    # Power Settings
    bs_max_power_dbm: float = 43.0
    ut_max_power_dbm: float = 23.0

    # Resource Grid (Inherited from HybridSimulationCommonConfig)
    # resource_grid field is available.

    # Antenna Arrays
    # bs_array and ut_array are created inside the simulator based on config parameters.
    # We do not store the object instances here to avoid serialization issues and keep config pure data.
    # bs_array: Any = None
    # ut_array: Any = None

    # Scenario
    scenario: str = "umi"  # "umi", "uma", "rma"

    # Mobility/Link Adaptation
    coherence_time: int = 10  # slots
    pf_beta: float = 0.98

    # Results
    output_dir: str = "experiments/hybrid_beamforming/sls/results"
    mpr_table_path: str = "experiments/hybrid_beamforming/lls/results/mpr_table.csv"

    def __init__(self):
        super().__init__()
        # self.resource_grid = ResourceGridConfig(
        #     num_ofdm_symbols=1,
        #     fft_size=72,
        #     subcarrier_spacing=self.subcarrier_spacing / 1e3,
        #     pilot_ofdm_symbol_indices=[2, 11],
        # )

        # Instantiate Antenna Arrays from config
        self.bs_array = PanelArray(
            num_rows=self.bs_num_rows_panel,
            num_cols=self.bs_num_cols_panel,
            num_rows_per_panel=self.bs_num_rows_per_panel,
            num_cols_per_panel=self.bs_num_cols_per_panel,
            polarization=self.bs_polarization,
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.carrier_frequency,
        )
        self.ut_array = PanelArray(
            num_rows=self.ut_num_rows_panel,
            num_cols=self.ut_num_cols_panel,
            num_rows_per_panel=self.ut_num_rows_per_panel,
            num_cols_per_panel=self.ut_num_cols_per_panel,
            polarization=self.ut_polarization,
            polarization_type="cross",
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency,
        )
