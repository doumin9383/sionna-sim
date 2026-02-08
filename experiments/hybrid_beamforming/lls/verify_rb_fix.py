import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from experiments.hybrid_beamforming.lls.run_papr_sim import run_papr_simulation
from experiments.hybrid_beamforming.lls.my_configs import HybridLLSConfig
import matplotlib.pyplot as plt


def test_rb_variation():
    print("Testing LLS with different RB sizes...")

    # Create a lightweight config for testing
    config = HybridLLSConfig()
    config.batch_size = 10
    config.num_batches = 1
    config.min_total_samples = 10

    # Limit sweep
    config.waveforms = [{"name": "DFT-s-OFDM", "is_dft_s": True}]
    config.modulations = {"QPSK": 2}
    config.ranks = [1]

    # Use two distinct RB sizes to check if they run
    config.rb_counts = [6, 50]

    config.granularities = ["Wideband"]

    config.output_file = "experiments/hybrid_beamforming/lls/results/test_rb_fix.csv"

    try:
        run_papr_simulation(config)
        print("Simulation executed successfully.")

        # Check if output exists
        import pandas as pd

        if os.path.exists(config.output_file):
            df = pd.read_csv(config.output_file)
            print("Output CSV created.")
            print(df[["num_rb", "papr_db_99.9"]])

            # Verify we have results for both RB counts
            rbs = df["num_rb"].unique()
            if 6 in rbs and 50 in rbs:
                print("SUCCESS: Simulation ran for both RB sizes.")
            else:
                print(f"FAILURE: Missing RB sizes in output. Found: {rbs}")
        else:
            print("FAILURE: Output file not found.")

    except Exception as e:
        print(f"FAILURE: Execution crashed. {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_rb_variation()
