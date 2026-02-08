import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from experiments.hybrid_beamforming.lls.components.pusch_model import (
    PUSCHCommunicationModel,
)


def verify_papr_convergence():
    print("Verifying PAPR Convergence...")

    # Parameters for a typical scenario
    # DFTS-OFDM, Narrowband, Rank 1 (Most critical for PAPR)
    scenario = {
        "waveform": "DFT-s-OFDM",
        "transform_precoding": True,
        "modulation": "64QAM",  # Arbitrary
        "mcs_index": 20,
        "rank": 1,
        "num_rb": 50,
        "granularity": "Narrowband",
        "num_tx": 8,
    }

    batch_size = 100
    total_samples_list = [1000, 5000, 10000, 20000, 50000, 100000]

    papr_999_values = []

    # Initialize Model once to save time, or re-init if needed
    # Re-init to be safe and clean
    model = PUSCHCommunicationModel(
        carrier_frequency=3.5e9,
        subcarrier_spacing=30e3,
        num_tx_ant=scenario["num_tx"],
        num_rx_ant=scenario["num_tx"],
        num_layers=scenario["rank"],
        num_rb=scenario["num_rb"],
        enable_transform_precoding=scenario["transform_precoding"],
        mcs_index=scenario["mcs_index"],
        precoding_granularity=scenario["granularity"],
        papr_oversampling_factor=4,
    )

    # To accumulate samples efficiently, we can generate a large pool or do incremental
    # Let's do incremental to simulate "running more batches"

    all_papr_samples = []

    current_count = 0
    max_target = max(total_samples_list)

    pbar = tqdm(total=max_target, desc="Generating Samples")

    while current_count < max_target:
        # Generate batch
        x = model.transmitter(batch_size)
        papr_db_batch = model.compute_papr(x).numpy().flatten()

        all_papr_samples.extend(papr_db_batch)
        current_count += len(papr_db_batch)
        pbar.update(len(papr_db_batch))

    pbar.close()

    # Now calculate 99.9% for each subset
    print("\nCalculating 99.9% PAPR for different sample sizes:")
    all_papr_samples = np.array(all_papr_samples)

    for n in total_samples_list:
        if n > len(all_papr_samples):
            print(f"Warning: Not enough samples for {n}")
            continue

        subset = all_papr_samples[:n]
        sorted_subset = np.sort(subset)
        idx = int(0.999 * len(sorted_subset))
        papr_val = sorted_subset[idx]
        papr_999_values.append(papr_val)
        print(f"  Samples: {n:6d} -> PAPR 99.9%: {papr_val:.4f} dB")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(total_samples_list, papr_999_values, "o-", linewidth=2)
    plt.xlabel("Total Samples")
    plt.ylabel("PAPR 99.9% [dB]")
    plt.title(
        f"PAPR Convergence Analysis\n{scenario['waveform']}, {scenario['granularity']}, Rank {scenario['rank']}"
    )
    plt.grid(True)

    output_dir = "experiments/hybrid_beamforming/lls/results/verification"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "papr_convergence.png")
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    verify_papr_convergence()
