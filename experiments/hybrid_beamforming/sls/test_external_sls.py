import os
import sys
import tensorflow as tf

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

tf.config.run_functions_eagerly(True)

from experiments.hybrid_beamforming.sls.simulator import HybridSystemSimulator
from experiments.hybrid_beamforming.sls.my_configs import HybridSLSConfig
from experiments.hybrid_beamforming.sls.components.external_channel_loader import (
    ExternalChannelLoader,
)
from sionna.phy.channel.tr38901 import PanelArray
from sionna.phy.ofdm import ResourceGrid


def test_external_sim():
    print("Testing External Data SLS Pipeline...")

    # 1. Initialize External Loader
    loader = ExternalChannelLoader("data/processed/coverage.zarr")

    # 2. Configuration (Match the dummy data)
    carrier_frequency = 3.5e9

    bs_array = PanelArray(
        num_rows=1,
        num_cols=1,
        num_rows_per_panel=4,
        num_cols_per_panel=4,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="38.901",
        carrier_frequency=carrier_frequency,
    )
    ut_array = PanelArray(
        num_rows=1,
        num_cols=1,
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
    )

    # Create ResourceGrid
    rg = ResourceGrid(
        num_ofdm_symbols=1,
        fft_size=24,
        subcarrier_spacing=30e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=6,
    )

    sim_config = HybridSLSConfig(
        scenario="uma",
        num_rings=1,  # num_rings=1 -> 1-ring = 21 sectors
        num_ut_per_sector=2,  # Total num_ut = 42
        num_neighbors=4,
        use_rbg_granularity=True,
        rbg_size_rb=1,
        bs_array=bs_array,
        ut_array=ut_array,
        resource_grid=rg,
        batch_size=1,
    )

    # 3. Instantiate Simulator with External Loader
    sim = HybridSystemSimulator(sim_config, external_loader=loader)

    # 4. Run Simulation
    print("Running simulation slot...")
    # num_slots=1, tx_power=30dBm
    sim(num_slots=1, tx_power_dbm=30.0)

    # 5. Report Memory
    try:
        mem = tf.config.experimental.get_memory_info("GPU:0")
        print(f"VRAM Usage (Peak): {mem['peak'] / 1024**2:.2f} MB")
        print(f"VRAM Usage (Current): {mem['current'] / 1024**2:.2f} MB")
    except:
        print(
            "Could not retrieve GPU memory info (maybe CPU run or unsupported TF version)"
        )

    print("Phase 3 Verification: Simulation finished successfully!")


if __name__ == "__main__":
    test_external_sim()
