#
# Copyright (c) 2024 Sionna Sim (Contributor)
#

import os
import sys

# Add project root to sys.path to allow importing 'experiments' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from experiments.sls_end2end_hybrid_beam.components.pusch_model import (
    PUSCHCommunicationModel,
)


def run_papr_simulation(
    output_file="experiments/sls_end2end_hybrid_beam/results/mpr_table.csv",
):

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_dir = os.path.dirname(output_file)

    # Simulation Parameters
    batch_size = 100  # Adjust based on GPU memory
    num_batches = 10  # Total samples = 1000 slots

    # Scenarios to sweep
    scenarios = []

    # Define Modulation to MCS Index mapping (approximate for Table 1)
    # QPSK: 2, 16QAM: 11, 64QAM: 20, 256QAM: 28 (Sample indices)
    modulations = {"QPSK": 2, "16QAM": 11, "64QAM": 20, "256QAM": 28}

    # 1. CP-OFDM
    for mod_name, mcs_idx in modulations.items():
        for rank in [1, 2, 4]:
            scenarios.append(
                {
                    "waveform": "CP-OFDM",
                    "transform_precoding": False,
                    "modulation": mod_name,
                    "mcs_index": mcs_idx,
                    "rank": rank,
                }
            )

    # 2. DFT-s-OFDM (Transform Precoding)
    # Usually Rank 1 only for DFT-s-OFDM in typical usage (though MIMO is possible in Rel 16+)
    # We stick to Rank 1 for now as per instructions "DFT-s-OFDMは通常Rank 1"
    for mod_name, mcs_idx in modulations.items():
        scenarios.append(
            {
                "waveform": "DFT-s-OFDM",
                "transform_precoding": True,
                "modulation": mod_name,
                "mcs_index": mcs_idx,
                "rank": 1,
            }
        )

    results = []
    all_papr_data = {}

    print(f"Starting PAPR Simulation with {len(scenarios)} scenarios...")

    for sc in tqdm(scenarios):
        # Scenario identifier for filenames
        scenario_id = f"{sc['waveform']}_{sc['modulation']}_Rank{sc['rank']}"

        # Instantiate Model
        num_tx = 4
        if sc["rank"] > num_tx:
            num_tx = sc["rank"]

        try:
            model = PUSCHCommunicationModel(
                carrier_frequency=3.5e9,
                subcarrier_spacing=30e3,
                num_tx_ant=num_tx,
                num_rx_ant=num_tx,
                num_layers=sc["rank"],
                enable_transform_precoding=sc["transform_precoding"],
                mcs_index=sc["mcs_index"],
                papr_oversampling_factor=4,
            )

            papr_values = []

            for i in range(num_batches):
                # Generate signal
                x = model.transmitter(batch_size)

                # Save a sample waveform for EVERY scenario (just the first batch)
                if i == 0:
                    plot_individual_waveform(x, scenario_id, results_dir)

                # Compute PAPR
                papr_db_batch = model.compute_papr(x)
                papr_values.extend(papr_db_batch.numpy().flatten())

            # Store for global comparison
            wave_mod_key = f"{sc['waveform']} ({sc['modulation']})"
            if wave_mod_key not in all_papr_data:
                all_papr_data[wave_mod_key] = []
            all_papr_data[wave_mod_key].extend(papr_values)

            # Compute and Plot individual CCDF
            papr_sorted = np.sort(papr_values)
            plot_individual_ccdf(papr_sorted, scenario_id, results_dir)

            # Compute 99.9% CCDF
            idx = int(0.999 * len(papr_sorted))
            papr_999 = papr_sorted[idx]

            # Record result
            res = sc.copy()
            res["papr_db_99.9"] = papr_999
            results.append(res)

        except Exception as e:
            print(f"Error in scenario {sc}: {e}")

    # Plot Comparison CCDF (Cleaned up)
    plot_summary_ccdf(all_papr_data, results_dir)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Simulation Complete. Results saved to {output_file}")


def plot_individual_waveform(x, scenario_id, results_dir):
    """Saves a plot of the time domain waveform for a specific scenario."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    # Take the first batch, first antenna
    # Plot a longer sequence to see the "flatness" across multiple symbols
    num_samples = min(x.shape[-1], 5000)
    sample = tf.abs(x[0, 0, :num_samples]).numpy()

    # Calculate RMS for reference
    rms = np.sqrt(np.mean(sample**2))

    plt.figure(figsize=(15, 5))
    plt.plot(sample, lw=0.8, label="Instantaneous Amplitude")
    plt.axhline(
        y=rms, color="r", linestyle="--", alpha=0.7, label=f"RMS Level ({rms:.2f})"
    )

    plt.title(f"Time Domain Amplitude (Long View): {scenario_id}")
    plt.xlabel("Time Samples")
    plt.ylabel("Absolute Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(results_dir, "waveforms", f"waveform_{scenario_id}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_individual_ccdf(papr_sorted, scenario_id, results_dir):
    """Saves a CCDF plot for a specific scenario."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    ccdf = 1.0 - np.arange(len(papr_sorted)) / float(len(papr_sorted))

    plt.figure(figsize=(8, 6))
    plt.semilogy(papr_sorted, ccdf, lw=2)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel("PAPR [dB]")
    plt.ylabel("CCDF (Prob > PAPR)")
    plt.title(f"PAPR CCDF: {scenario_id}")
    plt.ylim(1e-3, 1)
    plt.xlim(0, 15)

    save_path = os.path.join(results_dir, "ccdfs", f"ccdf_{scenario_id}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_summary_ccdf(all_papr_data, results_dir):
    """Saves a summary CCDF plot with all modulations compared."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    plt.figure(figsize=(12, 8))
    for label, values in all_papr_data.items():
        values = np.sort(values)
        ccdf = 1.0 - np.arange(len(values)) / float(len(values))
        plt.semilogy(values, ccdf, label=label, lw=2)

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel("PAPR [dB]")
    plt.ylabel("CCDF (Prob > PAPR)")
    plt.title("PAPR CCDF Summary Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim(1e-3, 1)
    plt.xlim(0, 15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "papr_ccdf_summary.png"))
    plt.close()


if __name__ == "__main__":
    # Actually checking path
    run_papr_simulation()
