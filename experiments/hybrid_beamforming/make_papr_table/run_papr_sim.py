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
from experiments.hybrid_beamforming.make_papr_table.configs import HybridLLSConfig
from experiments.hybrid_beamforming.make_papr_table.components.pusch_model import (
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
                    for strat in config.precoding_strategies:
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
                                    "strategy": strat,
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

        strat_str = "SVD" if sc["strategy"] == "SVD" else "ID"
        scenario_id = f"{sc['waveform']}_{sc['modulation']}_R{sc['rank']}_RB{sc['num_rb']}_{strat_str}_{gran_str}"

        # try:
        model = PUSCHCommunicationModel(
            config=config,
            num_layers=sc["rank"],
            enable_transform_precoding=sc["transform_precoding"],
            precoding_granularity=sc["granularity"],
            num_rb=sc["num_rb"],
            precoding_strategy=sc["strategy"],
        )

        papr_values = []
        cm_values = []

        for i in range(current_num_batches):
            # Generate signal
            x = model.transmitter(batch_size)

            # Save a sample waveform (only for a subset to avoid flooding disk)
            if i == 0 and sc["num_rb"] in representative_rb:
                plot_individual_waveform(x, scenario_id, results_dir)

            # Determine active antennas (Inactive Antenna Filtering)
            # Compute power per antenna [batch, tx]
            power = tf.reduce_mean(tf.abs(x) ** 2, axis=-1)
            threshold = 1e-6  # Power threshold for active antenna
            active_mask = power > threshold

            # Compute PAPR
            papr_db_batch = model.compute_papr(x)
            # Filter inactive antennas
            valid_papr = tf.boolean_mask(papr_db_batch, active_mask)
            papr_values.extend(valid_papr.numpy().flatten())

            # Compute CM
            cm_values_batch = compute_cm(x, active_mask)
            cm_values.extend(cm_values_batch.numpy().flatten())

        # Store for global comparison
        if sc["num_rb"] in representative_rb:
            # Use a structured key to allow parsing later: Waveform|Modulation|Rank|num_rb|Strategy|Granularity
            data_key = f"{sc['waveform']}|{sc['modulation']}|{sc['rank']}|{sc['num_rb']}|{sc['strategy']}|{sc['granularity']}"
            if data_key not in all_papr_data:
                all_papr_data[data_key] = []
            all_papr_data[data_key].extend(papr_values)

        # Compute and Plot individual CCDF (Selective)
        papr_sorted = np.sort(papr_values)
        if sc["num_rb"] in representative_rb:
            plot_individual_ccdf(papr_sorted, scenario_id, results_dir)

        # Compute 10e-3 CCDF and CM stats
        idx = int(0.999 * len(papr_sorted))
        papr_10e_3 = papr_sorted[idx] if len(papr_sorted) > 0 else 0.0

        # CM Calculation (Average CM is typical, but we can also store raw if needed)
        # For MPR table, we usually use a representative CM value.
        # 3GPP defines CM as a property of the signal, so mean or worst-case?
        # Typically the CM itself is a scalar metric per signal.
        # We have a distribution of CMs (one per slot/antenna).
        # We can take the mean CM or 99% CM?
        # 3GPP TS 38.101 usually refers to CM as a single value for a configuration.
        # Let's take the mean of the calculated CMs.
        if len(cm_values) > 0:
            cm_mean = np.mean(cm_values)
            cm_99 = np.percentile(cm_values, 99)
        else:
            cm_mean = 0.0
            cm_99 = 0.0

        # Record result
        res = sc.copy()
        res["papr_db_10e-3"] = papr_10e_3
        res["cm_db"] = cm_mean
        res["cm_db_99"] = cm_99
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


def compute_cm(x, mask=None):
    """
    Computes Cubic Metric (CM) of time-domain signal x.
    x: [batch, tx, time]
    mask: [batch, tx] boolean mask of active antennas. If None, all assumed active.
    """
    # Normalize signal
    # 3GPP defines CM for the signal. We calculate it per antenna active port.

    if mask is not None:
        # Mask x to get only active signals.
        valid_x = tf.boolean_mask(x, mask)  # [num_valid_samples, time]

        if tf.size(valid_x) == 0:
            return tf.constant([], dtype=tf.float32)

        # Normalize
        # Rms of each active signal
        rms = tf.sqrt(tf.reduce_mean(tf.abs(valid_x) ** 2, axis=-1, keepdims=True))
        x_norm = valid_x / tf.cast(rms + 1e-12, valid_x.dtype)

        # CM calculation
        # v_rms3 = sqrt(mean(|x|^6))
        v_pow6_mean = tf.reduce_mean(tf.abs(x_norm) ** 6, axis=-1)
        # cm = 20 * log10(v_rms3) / 1.56 = 10 * log10(v_pow6_mean) / 1.56
        cm_values = 10.0 * tf.math.log(v_pow6_mean) / tf.math.log(10.0) / 1.56

        return cm_values  # [num_valid_samples]

    else:
        # Standard calculation preserving shape
        rms = tf.sqrt(tf.reduce_mean(tf.abs(x) ** 2, axis=-1, keepdims=True))
        x_norm = x / tf.cast(rms + 1e-12, x.dtype)
        v_pow6_mean = tf.reduce_mean(tf.abs(x_norm) ** 6, axis=-1)
        cm_values = 10.0 * tf.math.log(v_pow6_mean) / tf.math.log(10.0) / 1.56
        return cm_values


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

        # Parse key: Waveform|Modulation|Rank|num_rb|Strategy|Granularity
        parts = key.split("|")
        # waveform = parts[0]
        modulation = parts[1]
        rank = int(parts[2])
        # num_rb = parts[3]
        strategy = parts[4]
        granularity = parts[5]

        # Determine styles
        color = rank_colors.get(rank, "black")

        ls = gran_styles.get(str(granularity), "-")

        # If strategy is Identity (Non-coherent), maybe use a different line style or marker?
        # Or just label it as "Non-coherent" in the legend?
        # Let's use a specific style for Non-coherent if needed, or just rely on Granularity/Rank.
        # User asked for "Non-coherent" category.
        # If strategy is Identity, we can override label or style.
        # Let's say: SVD -> solid/dashed based on gran. Identity -> maybe dashdot?

        if strategy == "Identity":
            ls = "-."
            granularity_label = "Non-coherent"
        else:
            granularity_label = granularity

        # specific check if granularity is "G...RB" string from previous logic or raw value
        # In this updated code we passed raw values in key
        # If it was an integer in config (e.g. 2, 4), handle it
        if granularity.isdigit():
            # If it's a number (RBG size), usually treat as Subband-like or separate style?
            # For now, let's treat numbers as dashed
            if strategy != "Identity":
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
            label=f"{modulation} R{rank} {strategy} {granularity}",  # internal label
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
