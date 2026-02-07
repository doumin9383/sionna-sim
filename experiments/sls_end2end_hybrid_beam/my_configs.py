from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
import numpy as np

# 基底クラスのインポート
from libs.my_configs import (
    ResourceGridConfig,
    PlanarArrayConfig,
)


@dataclass
class HybridSLSConfig:
    """System Level Simulation Configuration for sls_end2end_hybrid_beam"""

    # Simulation Control
    batch_size: int = 1
    num_rings: int = 1
    num_ut_per_sector: int = 1
    num_slots: int = 1

    # RF/Frequency
    carrier_frequency: float = 3.5e9
    subcarrier_spacing: float = 30e3

    # Power Settings
    bs_max_power_dbm: float = 43.0
    ut_max_power_dbm: float = 23.0

    # Resource Grid
    resource_grid: ResourceGridConfig = field(
        default_factory=lambda: ResourceGridConfig(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            cyclic_prefix_length=6,
        )
    )

    # Antenna Arrays
    bs_array: Any = None  # sionna.phy.channel.tr38901.PanelArray
    ut_array: Any = None  # sionna.phy.channel.tr38901.PanelArray

    # Scenario
    scenario: str = "uma"  # "umi", "uma", "rma"
    direction: str = "downlink"  # "uplink", "downlink"

    # Mobility/Link Adaptation
    coherence_time: int = 10  # slots
    pf_beta: float = 0.98

    # Results
    output_dir: str = "experiments/sls_end2end_hybrid_beam/results"


@dataclass
class LLSSweepConfig:
    """Link Level Simulation Sweep Configuration for run_papr_sim.py"""

    # Simulation Control
    batch_size: int = 100
    num_batches: int = 10

    # Sweep Parameters
    modulations: Dict[str, int] = field(
        default_factory=lambda: {"QPSK": 2, "16QAM": 11, "64QAM": 20, "256QAM": 28}
    )
    waveforms: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "CP-OFDM", "is_dft_s": False},
            {"name": "DFT-s-OFDM", "is_dft_s": True},
        ]
    )
    ranks: List[int] = field(default_factory=lambda: [1, 2, 4])
    rb_counts: List[int] = field(default_factory=lambda: [1, 20, 100])
    granularities: List[Union[int, str]] = field(
        default_factory=lambda: [2, "Wideband"]
    )

    # Physics Parameters
    carrier_frequency: float = 3.5e9
    subcarrier_spacing: float = 30e3
    papr_oversampling_factor: int = 4

    # Results
    output_file: str = "experiments/sls_end2end_hybrid_beam/results/mpr_table.csv"
