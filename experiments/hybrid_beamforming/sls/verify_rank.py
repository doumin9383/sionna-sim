import os
import sys

# Ensure current directory is in python path
sys.path.append(os.getcwd())

import tensorflow as tf
from experiments.hybrid_beamforming.sls.simulator import SystemSimulator
from experiments.hybrid_beamforming.sls.configs import SLSConfig


def main():
    print("Verifying SystemSimulator Rank Allocation with Batching...")

    # Configure GPU/Memory
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print(
        "Physical devices cannot be modified after being initialized check passed (or ignored)"
    )

    # Config with small batch size to trigger looping
    config = SLSConfig()
    config.batch_size = 1
    # HexGrid Topology settings
    config.num_rings = 1  # 7 sites = 21 cells
    config.num_ut_per_sector = 1
    # Total UTs = 21 * 1 = 21

    config.batch_size_ut = 5  # Should trigger 5 loops (21/5)

    config.carrier_frequency = 28e9
    config.bandwidth = 100e6
    config.num_ofdm_symbols = 14
    config.subcarrier_spacing = 30e3

    # Run Simulation
    try:
        print("Initializing Simulator...")
        sim = SystemSimulator(config)

        # Verify num_ut
        print(f"Num BS: {sim.num_bs}, Num UT: {sim.num_ut}")

        # Dummy inputs
        num_drops = 1
        # tx_power needs to match [batch, num_ut]
        tx_power_dbm = tf.fill([config.batch_size, sim.num_ut], 23.0)

        print("Starting Simulation Loop...")
        hist = sim(num_drops, tx_power_dbm)

        print("Simulation completed successfully!")

        if "rank" in hist:
            print("Rank shape:", hist["rank"].shape)
            print("Mean Rank:", tf.reduce_mean(hist["rank"]).numpy())

        if "num_decoded_bits" in hist:
            print("Throughput shape:", hist["num_decoded_bits"].shape)
            print("Mean TP:", tf.reduce_mean(hist["num_decoded_bits"]).numpy())

    except Exception as e:
        print(f"Simulation FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
