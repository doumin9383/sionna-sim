from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
from libs.my_configs import ResourceGridConfig, PlanarArrayConfig
from experiments.hybrid_beamforming.shared.configs import HybridSimulationCommonConfig


@dataclass
class HybridLLSConfig(HybridSimulationCommonConfig):
    """Link Level Simulation Configuration for experiments/hybrid_beamforming/lls"""

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

    # Physics Parameters (Inherited)
    # carrier_frequency, subcarrier_spacing

    papr_oversampling_factor: int = 4

    # Results
    output_file: str = "experiments/hybrid_beamforming/lls/results/mpr_table.csv"
