from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
import numpy as np

# 基底クラスのインポート
from libs.my_configs import (
    ResourceGridConfig,
    PlanarArrayConfig,
)
from experiments.hybrid_beamforming.shared.configs import HybridSimulationCommonConfig


@dataclass
class HybridSLSConfig(HybridSimulationCommonConfig):
    """System Level Simulation Configuration for experiments/hybrid_beamforming/sls"""

    # Simulation Control
    batch_size: int = 1  # Debug: 1
    num_rings: int = 1
    num_ut_per_sector: int = 1  # Production: 4
    num_slots: int = 2  # Debug: 2
    precoding_granularity: str = "Narrowband"  # "Narrowband", "Subband", "Wideband"
    use_rbg_granularity: bool = True  # If True, calculate channel only at RBG centers
    num_neighbors: int = (
        16  # For spatial masking: num BS per UT to calculate channel for
    )

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
