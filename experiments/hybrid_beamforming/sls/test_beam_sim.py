import tensorflow as tf
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from experiments.hybrid_beamforming.sls.simulator import HybridSystemSimulator
from experiments.hybrid_beamforming.sls.my_configs import HybridSLSConfig


def test_beam_selection_integration():
    print("Testing Beam Selection Integration...")

    # 1. Setup Config
    config = HybridSLSConfig()
    config.batch_size = 1
    config.num_ut_per_sector = 1  # Minimal setup
    config.num_ut_drops = 1
    config.num_neighbors = 3
    config.precoding_granularity = "Wideband"
    config.beam_selection_method = "subpanel_sweep"

    # 2. Instantiate Simulator
    sim = HybridSystemSimulator(config)
    print("Simulator instantiated.")

    # 3. Run Simulation Drop
    print("Running simulation drop...")
    try:
        hist = sim.call(num_drops=1, tx_power_dbm=config.bs_max_power_dbm)
        print("Simulation drop completed successfully!")

        # Verify results structure
        print("History keys:", hist.keys())

        # Check if throughput is non-zero (indication of working link)
        tput = hist["num_decoded_bits"]
        print("Throughput shape:", tput.shape)
        print("Throughput mean:", tf.reduce_mean(tput).numpy())

        # Check SINR
        sinr = hist["sinr_eff"]
        print("SINR shape:", sinr.shape)
        print("SINR mean:", tf.reduce_mean(sinr).numpy())

    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_beam_selection_integration()
