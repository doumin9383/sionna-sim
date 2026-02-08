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
from experiments.hybrid_beamforming.lls.my_configs import HybridLLSConfig
from experiments.hybrid_beamforming.lls.components.pusch_model import (
    PUSCHCommunicationModel,
)


def run_papr_simulation(config: HybridLLSConfig = HybridLLSConfig()):

    output_file = config.output_file
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_dir = os.path.dirname(output_file)

    # Simulation Parameters
    batch_size = config.batch_size
    num_batches = config.num_batches

    # Scenarios to sweep
    scenarios = []

    # Iterate through combinations from config
    for wf in config.waveforms:
        wf_name = wf["name"]
        is_dft_s = wf["is_dft_s"]
        for mod_name, mcs_idx in config.modulations.items():
            for rank in config.ranks:
                # DFT-s-OFDM is usually Rank 1
                if is_dft_s and rank > 1:
                    continue
                for num_rb in config.rb_counts:
                    for gran in config.granularities:
                        scenarios.append(
                            {
                                "waveform": wf_name,
                                "transform_precoding": is_dft_s,
                                "modulation": mod_name,
                                "mcs_index": mcs_idx,
                                "rank": rank,
                                "num_rb": num_rb,
                                "granularity": gran,
                            }
                        )

    results = []
    all_papr_data = {}

    print(f"Starting PAPR Simulation with {len(scenarios)} scenarios...")

    # For large sweeps, reduce batches if needed
    min_total_samples = config.min_total_samples
    # Calculate required batches to meet minimum samples
    required_batches = int(np.ceil(min_total_samples / batch_size))
    # Use the larger of config.num_batches or required_batches
    current_num_batches = max(num_batches, required_batches)

    print(
        f"Targeting {min_total_samples} samples. Batch size {batch_size} -> running {current_num_batches} batches."
    )

    for sc in tqdm(scenarios):
        # Scenario identifier for filenames
        # Shorten ID to avoid too long filenames
        gran_str = (
            f"G{sc['granularity']}" if isinstance(sc["granularity"], int) else "GWB"
        )
        scenario_id = f"{sc['waveform']}_{sc['modulation']}_R{sc['rank']}_RB{sc['num_rb']}_{gran_str}"

        # Instantiate Model
        # Antennas: Use 8 as base or same as rank
        num_tx = max(sc["rank"], 8)

        try:
            model = PUSCHCommunicationModel(
                carrier_frequency=config.carrier_frequency,
                subcarrier_spacing=config.subcarrier_spacing,
                num_tx_ant=num_tx,
                num_rx_ant=num_tx,
                num_layers=sc["rank"],
                num_rb=sc["num_rb"],
                enable_transform_precoding=sc["transform_precoding"],
                mcs_index=sc["mcs_index"],
                precoding_granularity=sc["granularity"],
                papr_oversampling_factor=config.papr_oversampling_factor,
            )

            papr_values = []

            for i in range(current_num_batches):
                # Generate signal
                x = model.transmitter(batch_size)

                # Save a sample waveform (only for a subset to avoid flooding disk)
                if i == 0 and sc["num_rb"] == 100 and sc["rank"] == 1:
                    plot_individual_waveform(x, scenario_id, results_dir)

                # Compute PAPR
                papr_db_batch = model.compute_papr(x)
                papr_values.extend(papr_db_batch.numpy().flatten())

            # Store for global comparison (Selective labels to avoid cluttered legend)
            if sc["num_rb"] == 50 and sc["granularity"] == "Wideband":
                wave_mod_key = f"{sc['waveform']} ({sc['modulation']}) R{sc['rank']}"
                if wave_mod_key not in all_papr_data:
                    all_papr_data[wave_mod_key] = []
                all_papr_data[wave_mod_key].extend(papr_values)

            # Compute and Plot individual CCDF (Selective)
            papr_sorted = np.sort(papr_values)
            if sc["num_rb"] == 100:
                plot_individual_ccdf(papr_sorted, scenario_id, results_dir)

            # Compute 99.9% CCDF
            idx = int(0.999 * len(papr_sorted))
            papr_999 = papr_sorted[idx]

            # Record result
            res = sc.copy()
            res["papr_db_99.9"] = papr_999
            results.append(res)

            # --- Memory Management ---
            # Important for large sweeps on limited VRAM
            del model
            tf.keras.backend.clear_session()
            import gc

            gc.collect()

        except Exception as e:
            print(f"Error in scenario {sc}: {e}")
            tf.keras.backend.clear_session()

    # Plot Comparison CCDF (Cleaned up)
    plot_summary_ccdf(all_papr_data, results_dir)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Simulation Complete. Results saved to {output_file}")


def plot_individual_waveform(x, scenario_id, results_dir):
    """Saves a plot of the time domain waveform with subplots for antennas."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    # x shape: [batch, tx, time]
    # Plot enough samples to see one whole slot or a significant part of it
    # 15000 samples is usually ~4 OFDM symbols at typical SCS/FFT
    num_samples = min(x.shape[-1], 15000)

    num_ant_to_plot = min(x.shape[1], 4)
    fig, axes = plt.subplots(
        num_ant_to_plot, 1, figsize=(15, 3 * num_ant_to_plot), sharex=True
    )
    if num_ant_to_plot == 1:
        axes = [axes]

    for i in range(num_ant_to_plot):
        sample = tf.abs(x[0, i, :num_samples]).numpy()
        rms = np.sqrt(np.mean(sample**2))

        ax = axes[i]
        ax.plot(sample, lw=0.6, label=f"Ant {i} Amplitude")
        ax.axhline(
            y=rms, color="r", linestyle="--", alpha=0.6, label=f"RMS ({rms:.2f})"
        )

        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f"Time Domain Waveform (Multi-Antenna): {scenario_id}")

    axes[-1].set_xlabel("Time Samples")
    plt.tight_layout()

    save_path = os.path.join(results_dir, "waveforms", f"waveform_{scenario_id}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


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
