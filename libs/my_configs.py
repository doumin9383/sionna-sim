from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
import numpy as np

# Sionna imports for type hints (mocked or real depending on environment)
# In a real environment, you would import these properly if available for type checking.
# from sionna.rt import Scene, PlanarArray, Transmitter, Receiver, Camera
# from sionna.phy.mimo import StreamManagement
# from sionna.phy.ofdm import ResourceGrid
# from sionna.phy.nr import CarrierConfig

@dataclass
class SceneConfig:
    """Configuration for sionna.rt.Scene"""
    # Using 'filename' instead of loading from file inside config,
    # but strictly mirroring __init__ might imply passing the loaded scene.
    # Here we define parameters to *load* or *configure* the scene.
    filename: Optional[str] = None
    frequency: float = 3.5e9 # Hz
    synthetic_array: bool = False
    dtype: Any = "complex64" # tf.complex64

@dataclass
class PlanarArrayConfig:
    """Configuration for sionna.rt.PlanarArray"""
    num_rows: int = 1
    num_cols: int = 1
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5
    pattern: str = "iso"
    polarization: str = "VH"


@dataclass
class TransmitterConfig:
    """Configuration for sionna.rt.Transmitter"""
    name: str
    position: List[float] # [x, y, z]
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # [yaw, pitch, roll]

    # In Sionna, you pass an AntennaArray instance to Transmitter.
    # Here we nest the config to create it.
    antenna_array: PlanarArrayConfig = field(default_factory=PlanarArrayConfig)

@dataclass
class ReceiverConfig:
    """Configuration for sionna.rt.Receiver"""
    name: str
    position: List[float] # [x, y, z]
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    antenna_array: PlanarArrayConfig = field(default_factory=PlanarArrayConfig)

@dataclass
class RayTracingConfig:
    """Parameters for PathSolver.__call__"""
    max_depth: int = 10
    samples_per_src: int = 1000000
    max_num_paths_per_src: int = 1000000
    synthetic_array: bool = True
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    seed: int = 42

# --- Logical Layer Configs (PHY) ---

@dataclass
class StreamManagementConfig:
    """Configuration for sionna.phy.mimo.StreamManagement"""
    rx_tx_association: np.ndarray # matrix of 0s and 1s
    num_streams_per_tx: int = 1

@dataclass
class ResourceGridConfig:
    """Configuration for sionna.phy.ofdm.ResourceGrid"""
    num_ofdm_symbols: int = 14
    fft_size: int = 64
    subcarrier_spacing: float = 30e3
    num_tx: int = 1
    num_streams_per_tx: int = 1
    cyclic_prefix_length: int = 0
    pilot_pattern: Optional[str] = "kronecker"
    pilot_ofdm_symbol_indices: Optional[List[int]] = field(default_factory=lambda: [2, 11])

@dataclass
class CarrierConfigConfig: # Naming collision potential, keeping consistent with Sionna class name suffix
    """Configuration for sionna.phy.nr.CarrierConfig"""
    n_cell_id: int = 1
    subcarrier_spacing: float = 30
    cyclic_prefix: str = "normal"
    # ... add other parameters as needed from Sionna NR

# --- Master Config ---

@dataclass
class SimulationConfig:
    """Master configuration for the simulation run"""
    scene: SceneConfig
    transmitters: List[TransmitterConfig]
    receivers: List[ReceiverConfig]
    ray_tracing: RayTracingConfig

    # Optional PHY layer configs
    image_filename: Optional[str] = None # For saving coverage map etc.
    output_dir: str = "results"
