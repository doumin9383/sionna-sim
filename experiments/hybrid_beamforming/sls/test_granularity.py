import tensorflow as tf
import os
import sys

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from experiments.hybrid_beamforming.sls.my_configs import HybridSLSConfig
from experiments.hybrid_beamforming.sls.simulator import HybridSystemSimulator
from sionna.phy.channel.tr38901 import PanelArray
from sionna.phy.ofdm import ResourceGrid


def test_granularity():
    # 1. Setup Antenna Arrays
    carrier_frequency = 3.5e9
    bs_array = PanelArray(
        num_rows=1,
        num_cols=1,
        num_rows_per_panel=2,
        num_cols_per_panel=2,
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

    # 2. Setup Resource Grid
    rg = ResourceGrid(
        num_ofdm_symbols=1,
        fft_size=12,  # 1 RB
        subcarrier_spacing=30e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=0,
    )

    # 3. Basic Config
    config = HybridSLSConfig(
        batch_size=1,
        num_rings=1,
        num_ut_per_sector=1,
        num_slots=2,
        precoding_granularity="Subband",
        rbg_size_rb=1,  # 1 RB = 12 subcarriers
        bs_array=bs_array,
        ut_array=ut_array,
        carrier_frequency=carrier_frequency,
        resource_grid=rg,
    )

    # Instantiate Simulator
    sim = HybridSystemSimulator(config)

    print("Running simulation with Subband (4RB)...")
    # Run
    throughput = sim.call(num_slots=2, tx_power_dbm=23.0)
    print("Throughput shape:", throughput.shape)
    print("Simulation finished successfully.")

    # Test Wideband
    print("\nRunning simulation with Wideband...")
    config.precoding_granularity = "Wideband"
    sim_wb = HybridSystemSimulator(config)
    throughput_wb = sim_wb.call(num_slots=2, tx_power_dbm=23.0)
    print("Throughput shape:", throughput_wb.shape)
    print("Wideband Simulation finished successfully.")

    # Test Narrowband
    print("\nRunning simulation with Narrowband...")
    config.precoding_granularity = "Narrowband"
    sim_nb = HybridSystemSimulator(config)
    throughput_nb = sim_nb.call(num_slots=2, tx_power_dbm=23.0)
    print("Throughput shape:", throughput_nb.shape)
    print("Narrowband Simulation finished successfully.")


if __name__ == "__main__":
    test_granularity()
