from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
import numpy as np

# 基底クラスのインポート
from wsim.rt.configs import (
    ResourceGridConfig,
    PlanarArrayConfig,
)
from experiments.hybrid_beamforming.global_config import SimulationCommonConfig

from sionna.phy.channel.tr38901 import PanelArray


@dataclass
class SLSConfig(SimulationCommonConfig):
    """System Level Simulation Configuration for experiments/hybrid_beamforming/sls"""

    # Simulation Control
    batch_size: int = 1  # Debug: 1

    num_ut_drops: int = 1  # Debug: 1. Number of random topology drops.
    num_slots: int = 1  # Fixed to 1 for snapshot simulation

    precoding_granularity: str = "Wideband"  # "Narrowband", "Subband", "Wideband"
    # precoding_strategy: str = "SVD"  # "SVD", "Identity" (Non-coherent)
    precoding_strategy: str = "Identity"  # "SVD", "Identity" (Non-coherent)
    use_rbg_granularity: bool = True  # If True, calculate channel only at RBG centers
    # waveform: str = "CP-OFDM"  # "CP-OFDM" or "DFT-s-OFDM"
    waveform: str = "DFT-s-OFDM"  # "CP-OFDM" or "DFT-s-OFDM"
    cyclic_prefix_length: int = 0  # CP length for ResourceGrid
    export_detailed_logs: bool = False  # If True, export detailed rank selection logs
    num_neighbors: Optional[int] = (
        # 4  # For spatial masking: num BS per UT to calculate channel for. None means all cells.
        None
    )
    num_layers: int = 4  # Number of layers for spatial multiplexing
    available_layers: List[int] = field(
        default_factory=lambda: [1, 2, 4]
    )  # Ranks to sweep in link adaptation
    batch_size_ut: int = 1  # UT batch size for memory management
    max_la_iterations: int = 2  # Max iterations for MPR/LA convergence
    max_rank_selection_iterations: int = 2  # Max iterations for Rank Selection loop
    num_symbols_per_slot: int = 14  # Number of symbols in a slot for throughput scaling
    num_data_symbols: int = 12  # Number of effective data symbols (excluding DMRS)

    # MPR Model
    mpr_model_type: str = "linear"  # "linear" or "table"
    mpr_linear_slope: float = 0.5
    mpr_linear_ref_papr: float = 9.6
    mpr_linear_ref_backoff: float = 0.5
    mpr_table_mode_column: str = "cm_db"

    # Beam Management
    beambook_oversampling_factor: int = 4
    beam_selection_method: str = (
        "subpanel_sweep"  # "subpanel_sweep" or "full_array_sweep"
    )

    num_rb = 66
    num_subcarriers = num_rb * 12

    # Power Settings
    bs_max_power_dbm: float = 43.0
    ut_max_power_dbm: float = 23.0
    power_control_method: str = "sionna"  # "sionna" or "custom"

    # Topology
    topology_type: str = "HexGrid"  # "HexGrid", "Custom", etc.
    topology_wrap: bool = True  # wrapネットワーク干渉を考慮するか
    # hex gridの場合
    num_rings: int = 1
    num_ut_per_sector: int = 1
    min_bs_ut_dist: float = 10.0  # Min distance between BS and UT
    max_bs_ut_dist: Optional[float] = (
        None  # Max distance, None means infinite/cell edge
    )

    @property
    def force_tx_identity(self):
        return self.precoding_strategy == "Identity"

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

    # Scenario
    scenario: str = "uma"  # "umi", "uma", "rma"

    # Mobility/Link Adaptation
    coherence_time: int = 10  # slots
    pf_beta: float = 0.98

    # Results
    output_dir: str = "experiments/hybrid_beamforming/sls/results"
    mpr_table_path: str = (
        "experiments/hybrid_beamforming/make_papr_table/results/papr_table.csv"
    )
    external_data_path: Optional[str] = None  # Path to external HDF5/Zarr data

    def __post_init__(self):
        super().__post_init__()

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
