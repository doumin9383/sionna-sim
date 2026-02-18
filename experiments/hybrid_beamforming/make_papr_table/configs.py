from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
from wsim.rt.configs import ResourceGridConfig, PlanarArrayConfig
from experiments.hybrid_beamforming.global_config import (
    SimulationCommonConfig,
    SYSTEM_MCS_INDICES,
    SYSTEM_WAVEFORMS,
)

import numpy as np


@dataclass
class HybridLLSConfig(SimulationCommonConfig):
    """Link Level Simulation Configuration for experiments/hybrid_beamforming/lls"""

    def __post_init__(self):
        super().__post_init__()

    # Simulation Control
    batch_size: int = 100
    num_batches: int = 5
    min_total_samples: int = 100

    # Sweep Parameters
    modulations: Dict[str, int] = field(
        default_factory=lambda: SYSTEM_MCS_INDICES.copy()
    )
    waveforms: List[Dict[str, Any]] = field(
        default_factory=lambda: SYSTEM_WAVEFORMS.copy()
    )
    ranks: List[int] = field(default_factory=lambda: [1, 2, 4])
    # rb_counts: List[int] = field(default_factory=lambda: np.arange(6, 66 + 6, 6))
    # rb_counts: List[int] = field(default_factory=lambda: np.arange(6, 132 + 6, 6))
    rb_counts: List[int] = field(default_factory=lambda: [6, 66, 132])
    granularities: List[Union[int, str]] = field(
        # default_factory=lambda: ["Subcarrer-wise", "Narrowband", "Subband", "Wideband"]
        default_factory=lambda: ["Narrowband", "Subband", "Wideband"]
    )
    precoding_strategies: List[str] = field(default_factory=lambda: ["SVD", "Identity"])

    papr_oversampling_factor: int = 4

    # Results
    output_file: str = (
        "experiments/hybrid_beamforming/make_papr_table/results/papr_table.csv"
    )
