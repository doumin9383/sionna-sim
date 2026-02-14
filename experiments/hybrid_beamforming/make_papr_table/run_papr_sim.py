#
# Copyright (c) 2024 Sionna Sim (Contributor)
#

import os
import sys

# Add project root to sys.path to allow importing 'experiments' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    # sys.path.append(project_root)

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from experiments.hybrid_beamforming.lls.configs import HybridLLSConfig
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

    # Choose a representative RB count for CCFD summary plot (e.g., middle of the sweep)
    # representative_rb = config.rb_counts[len(config.rb_counts) // 2]
    representative_rb = config.rb_counts

    for sc in tqdm(scenarios):
        # Scenario identifier for filenames
        # Shorten ID to avoid too long filenames
        if sc["granularity"] == "Subcarrer-wise":
            gran_str = "GSC"
        elif sc["granularity"] == "Narrowband":
            gran_str = "GNB"
        elif sc["granularity"] == "Subband":
            gran_str = "GSB"
        else:
            gran_str = "GWB"
        scenario_id = f"{sc['waveform']}_{sc['modulation']}_R{sc['rank']}_RB{sc['num_rb']}_{gran_str}"

        # try:
        model = PUSCHCommunicationModel(
            config=config,
            num_layers=sc["rank"],
            enable_transform_precoding=sc["transform_precoding"],
            precoding_granularity=sc["granularity"],
            num_rb=sc["num_rb"],
        )

        papr_values = []

        for i in range(current_num_batches):
            # Generate signal
            x = model.transmitter(batch_size)

            # Save a sample waveform (only for a subset to avoid flooding disk)
            if i == 0 and sc["num_rb"] in representative_rb:
                plot_individual_waveform(x, scenario_id, results_dir)

            # Compute PAPR
            papr_db_batch = model.compute_papr(x)
            papr_values.extend(papr_db_batch.numpy().flatten())

        # Store for global comparison
        if sc["num_rb"] in representative_rb:
            # Use a structured key to allow parsing later: Waveform|Modulation|Rank|num_rb|Granularity
            data_key = f"{sc['waveform']}|{sc['modulation']}|{sc['rank']}|{sc['num_rb']}|{sc['granularity']}"
            if data_key not in all_papr_data:
                all_papr_data[data_key] = []
            all_papr_data[data_key].extend(papr_values)

        # Compute and Plot individual CCDF (Selective)
        papr_sorted = np.sort(papr_values)
        if sc["num_rb"] in representative_rb:
            plot_individual_ccdf(papr_sorted, scenario_id, results_dir)

        # Compute 10e-3 CCDF
        idx = int(0.999 * len(papr_sorted))
        papr_10e_3 = papr_sorted[idx]

        # Record result
        res = sc.copy()
        res["papr_db_10e-3"] = papr_10e_3
        results.append(res)

        # --- Memory Management ---
        # Important for large sweeps on limited VRAM
        del model
        tf.keras.backend.clear_session()
        import gc

        gc.collect()

    # Plot Comparison CCDF (Cleaned up)
    plot_summary_ccdf(all_papr_data, results_dir)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Simulation Complete. Results saved to {output_file}")

    # Save raw data to .npz
    npz_file = output_file.replace(".csv", ".npz")
    np.savez_compressed(npz_file, **all_papr_data)
    print(f"Raw PAPR data saved to {npz_file}")


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
    import matplotlib.font_manager as fm
    from matplotlib.lines import Line2D

    plt.switch_backend("Agg")

    # Try to set Japanese font
    # Common Japanese fonts on Linux
    font_candidates = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAGothic",
        "IPAexGothic",
        "TakaoGothic",
        "VL Gothic",
    ]
    found_font = None
    for f in font_candidates:
        try:
            # Check if font is available
            f_prop = fm.FontProperties(fname=fm.findfont(f))
            if f_prop.get_name():
                plt.rcParams["font.family"] = f
                found_font = f
                break
        except:
            continue

    if found_font:
        print(f"Using Japanese font: {found_font}")
    else:
        print("Warning: No Japanese font found. Labels may not display correctly.")

    plt.figure(figsize=(12, 8))

    # Styling definitions
    rank_colors = {
        1: "#1f77b4",  # Blue
        2: "#ff7f0e",  # Orange
        4: "#2ca02c",  # Green
        8: "#d62728",  # Red
    }

    gran_styles = {
        "Wideband": "-",  # Solid
        "Subband": "--",  # Dashed
        "Narrowband": ":",  # Dotted
    }
    # Fallback for integer granularity if used

    mod_markers = {
        "QPSK": "o",
        "16QAM": "^",
        "64QAM": "s",
        "256QAM": "D",
    }

    # Sorting keys for consistent plotting order
    sorted_keys = sorted(all_papr_data.keys())

    for key in sorted_keys:
        values = np.sort(all_papr_data[key])

        # Parse key: Waveform|Modulation|Rank|num_rb|Granularity
        parts = key.split("|")
        # waveform = parts[0]
        modulation = parts[1]
        rank = int(parts[2])
        # num_rb = parts[3]
        granularity = parts[4]  # String "Wideband", "Subband", etc. or int

        # Determine styles
        color = rank_colors.get(rank, "black")

        ls = gran_styles.get(str(granularity), "-")
        # specific check if granularity is "G...RB" string from previous logic or raw value
        # In this updated code we passed raw values in key
        # If it was an integer in config (e.g. 2, 4), handle it
        if granularity.isdigit():
            # If it's a number (RBG size), usually treat as Subband-like or separate style?
            # For now, let's treat numbers as dashed
            ls = "--"

        marker = mod_markers.get(modulation, "x")

        ccdf = 1.0 - np.arange(len(values)) / float(len(values))

        # Plot line
        plt.semilogy(
            values,
            ccdf,
            color=color,
            linestyle=ls,
            linewidth=2,
            marker=marker,
            markevery=0.1,  # Show marker every 10% of points to avoid clutter
            markersize=6,
            label=f"{modulation} R{rank} {granularity}",  # internal label, not used for custom legend
        )

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel("PAPR [dB]")
    plt.ylabel("CCDF (確率 > PAPR)")
    plt.title("PAPR CCDF サマリ比較")

    plt.ylim(1e-3, 1)
    plt.xlim(0, 15)

    # --- Custom Legend ---
    # Rank Legend
    rank_handles = [
        Line2D([0], [0], color=c, lw=2, label=f"Rank {r}")
        for r, c in rank_colors.items()
    ]
    rank_legend = plt.legend(
        handles=rank_handles,
        title="Rank",
        loc="upper right",
        bbox_to_anchor=(1.15, 1.0),
    )
    plt.gca().add_artist(rank_legend)

    # Granularity Legend
    gran_handles = [
        Line2D([0], [0], color="gray", linestyle=ls, lw=2, label=g)
        for g, ls in gran_styles.items()
    ]
    gran_legend = plt.legend(
        handles=gran_handles,
        title="Granularity",
        loc="upper right",
        bbox_to_anchor=(1.15, 0.8),
    )
    plt.gca().add_artist(gran_legend)

    # Modulation Legend
    mod_handles = [
        Line2D([0], [0], color="gray", marker=m, linestyle="None", label=mod)
        for mod, m in mod_markers.items()
    ]
    mod_legend = plt.legend(
        handles=mod_handles,
        title="Modulation",
        loc="upper right",
        bbox_to_anchor=(1.15, 0.6),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "papr_ccdf_summary.png"))
    plt.close()


if __name__ == "__main__":
    # Actually checking path
    run_papr_simulation()
