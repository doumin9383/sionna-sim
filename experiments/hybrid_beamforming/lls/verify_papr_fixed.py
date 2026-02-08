import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from experiments.hybrid_beamforming.lls.my_configs import HybridLLSConfig
from experiments.hybrid_beamforming.lls.components.pusch_model import (
    PUSCHCommunicationModel,
)


def verify_papr_fix():
    output_dir = os.path.join(os.path.dirname(__file__), "results", "verification")
    os.makedirs(output_dir, exist_ok=True)

    # Config
    config = HybridLLSConfig()
    config.batch_size = 100  # Enough for smooth CCDF
    config.papr_oversampling_factor = 4  # Ensure high accuracy

    # Scenarios to test
    # We focus on Rank 1, 66 RBs (Wideband ish)
    rank = 1
    num_rb = 66
    granularity = "Wideband"

    scenarios = [
        {"name": "CP-OFDM", "df_s": False},
        {"name": "DFT-s-OFDM", "df_s": True},
    ]

    plt.figure(figsize=(10, 7))

    first_dfs_done = False

    for sc in scenarios:
        print(f"Running Scenario: {sc['name']}...")

        # Initialize Model
        # Note: We must manually set the RB count because the config object doesn't carry the loop variable
        model = PUSCHCommunicationModel(
            config=config,
            num_layers=rank,
            enable_transform_precoding=sc["df_s"],
            precoding_granularity=granularity,
        )

        # MANUALLY UPDATE PUSCH CONFIG RB COUNT
        # This fixes a potential bug where RB count wasn't updating
        model.pusch_config.n_size_bwp = num_rb
        model.pusch_config.n_start_bwp = 0
        model.pusch_config.check_config()

        # Generate Signal
        # Run once to initialize everything (Sionna might be lazy)
        x = model.transmitter(config.batch_size)

        # 1. Compute PDF (Original / DMRS Included)
        papr_full = model.compute_papr(x, exclude_dmrs=False)
        papr_full_vals = papr_full.numpy().flatten()

        # 2. Compute PAPR (No DMRS)
        papr_no_dmrs = model.compute_papr(x, exclude_dmrs=True)
        papr_no_dmrs_vals = papr_no_dmrs.numpy().flatten()

        # Plot CCDF
        label = sc["name"]

        # Full
        sorted_full = np.sort(papr_full_vals)
        ccdf_full = 1.0 - np.arange(len(sorted_full)) / len(sorted_full)
        plt.semilogy(
            sorted_full, ccdf_full, label=f"{label} (Full)", linestyle="--", alpha=0.7
        )

        # No DMRS
        sorted_no = np.sort(papr_no_dmrs_vals)
        ccdf_no = 1.0 - np.arange(len(sorted_no)) / len(sorted_no)
        # Use solid line for the "Fix" target (DFT-s No DMRS)
        ls = "-"
        # Make DFT-s-OFDM (No DMRS) stand out
        lw = 2.5 if sc["df_s"] else 1.5
        plt.semilogy(
            sorted_no, ccdf_no, label=f"{label} (No DMRS)", linestyle=ls, linewidth=lw
        )

        # Plot Waveform for one batch element (DFT-s-OFDM case)
        if sc["df_s"] and not first_dfs_done:
            plot_waveform_comparison(
                model, x, output_dir, f"waveform_dfs_{num_rb}rb.png"
            )
            first_dfs_done = True

    plt.xlabel("PAPR [dB]")
    plt.ylabel("CCDF")
    plt.title(f"PAPR Comparison: DMRS Exclusion Effect (Rank {rank}, {num_rb} RBs)")
    plt.grid(True, which="both", alpha=0.5)
    plt.legend()
    plt.ylim(1e-4, 1)
    plt.xlim(0, 14)

    save_path = os.path.join(output_dir, "papr_fix_verification.png")
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.close()  # Close figure


def plot_waveform_comparison(model, x, output_dir, filename):
    """
    Plots the time domain magnitude of the signal, highlighting DMRS vs Data symbols.
    """
    import matplotlib.pyplot as plt

    # Take first batch, first antenna
    # Shape [batch, tx, time]
    # Use index 0, 0
    sig = x[0, 0, :]  # [time]

    # Re-apply the logic to find DMRS indices
    dmrs_indices = model.pusch_config.dmrs_symbol_indices
    rg = model.transmitter.resource_grid
    fft_size = rg.fft_size
    cp_len = rg.cyclic_prefix_length
    num_symbols = rg.num_ofdm_symbols

    # Handle cp_len
    try:
        if isinstance(cp_len, int) or np.isscalar(cp_len):
            cp_lens = [int(cp_len)] * num_symbols
        else:
            cp_lens = list(cp_len)
    except:
        cp_lens = [int(cp_len)] * num_symbols

    # Create mask for DMRS
    is_dmrs = np.zeros(sig.shape[0], dtype=bool)
    current_idx = 0

    for i in range(num_symbols):
        sym_len = fft_size + cp_lens[i]
        start = current_idx
        end = current_idx + sym_len

        if i in dmrs_indices:
            if end > len(is_dmrs):
                break
            is_dmrs[start:end] = True

        current_idx = end

    # Plot
    plt.figure(figsize=(15, 6))
    t = np.arange(len(sig))
    mag = np.abs(sig)

    # Plot entire signal
    plt.plot(t, mag, label="Signal Magnitude", color="blue", alpha=0.6, linewidth=0.8)

    # Highlight DMRS
    # We create a boolean mask for plot
    # To make it visible, we can plot the DMRS part in red on top
    dmrs_mag = np.copy(mag)
    dmrs_mag[~is_dmrs] = np.nan
    plt.plot(t, dmrs_mag, color="red", alpha=0.8, linewidth=1.0, label="DMRS Region")

    # Highlight Data
    # data_mag = np.copy(mag)
    # data_mag[is_dmrs] = np.nan
    # plt.plot(t, data_mag, color='green', alpha=0.9, linewidth=1.0, label="Data Region")

    plt.xlabel("Time Samples")
    plt.ylabel("|x(t)|")
    plt.title("DFT-s-OFDM Time Domain Waveform (Red = DMRS)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    print(f"Saved waveform plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    verify_papr_fix()
