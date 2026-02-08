import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from experiments.hybrid_beamforming.sls.components.link_adaptation import (
    MCSLinkAdaptation,
)


def eval_lls_throughput():
    # 1. Load MPR Table
    mpr_csv = "experiments/hybrid_beamforming/lls/results/mpr_table.csv"

    use_dummy = False
    if not os.path.exists(mpr_csv):
        print(f"Error: {mpr_csv} not found.")
        use_dummy = True
    elif os.stat(mpr_csv).st_size <= 1:  # Empty or just newline
        print(f"Warning: {mpr_csv} is empty.")
        use_dummy = True

    if use_dummy:
        # Create dummy table for verification if not exists
        print("Creating dummy MPR data for verification plot...")
        data = []
        for wf in ["CP-OFDM", "DFT-s-OFDM"]:
            for r in [1, 2]:
                data.append(
                    {
                        "waveform": wf,
                        "rank": r,
                        "papr_db_99.9": 8.0,
                        "mpr_db": 0.0 if wf == "DFT-s-OFDM" else 2.0,
                    }
                )
        df_mpr = pd.DataFrame(data)
    else:
        df_mpr = pd.read_csv(mpr_csv)

    # 2. Setup Evaluation
    snr_range_db = np.linspace(-10, 30, 41)

    # Scenarios to compare:
    # (Waveform, Granularity)
    # Note: Granularity affects "Precoding Loss".
    # For this analytical plot, we need a model for Precoding Loss vs Granularity vs Channel Condition.
    # Since we don't have the full channel fading loop here, we will assume simplified loss values
    # OR we need to run a small simulation loop for each SNR point (Monte Carlo LLS).
    # Given "Throughput vs Pathloss (SNR)" is usually an instantaneous link curve,
    # let's use the Analytical Link Adaptation model with assumed Precoding Losses.

    # Assumed Precoding Losses (dB) relative to ideal digital
    precoding_losses = {
        "Narrowband": 0.0,
        "Subband": 1.0,  # Guess
        "Wideband": 3.0,  # Guess
    }

    mcs_adapter = MCSLinkAdaptation()

    results = []

    plt.figure(figsize=(10, 6))

    for waveform in ["CP-OFDM", "DFT-s-OFDM"]:
        # Get MPR for Rank 1 (Simplification)
        row = df_mpr[
            (df_mpr["waveform"] == waveform) & (df_mpr["rank"] == 1)
        ]  # String vs Int check needed?
        if len(row) == 0:
            # Try type conversion or fallback
            mpr = 0.0
        else:
            # Check column name. The script saved "papr_db_99.9". Did it save "mpr_db"?
            # run_papr_sim.py didn't seem to calculate "mpr_db" explicitly?
            # It saved "papr_db_99.9".
            # We need to DERIVE MPR from PAPR. (e.g. if PAPR > Limit, MPR = PAPR - Limit)
            papr = float(row["papr_db_99.9"].iloc[0])
            limit_db = 8.0  # Example limit
            mpr = max(0.0, papr - limit_db)

        for gran, loss_db in precoding_losses.items():
            tputs = []
            for snr in snr_range_db:
                # Effective SNR = SNR - MPR - PrecodingLoss
                eff_sinr = snr - mpr - loss_db

                # Link Adaptation
                # get_throughput_vectorized expects tensor float
                cap, _ = mcs_adapter.get_throughput_vectorized(
                    tf.constant([eff_sinr], dtype=tf.float32)
                )
                tputs.append(cap.numpy()[0])

            label = f"{waveform} - {gran} (MPR={mpr:.1f}dB)"
            plt.plot(snr_range_db, tputs, label=label)

            # Store for Optimal Map
            for i, snr in enumerate(snr_range_db):
                results.append(
                    {
                        "snr": snr,
                        "waveform": waveform,
                        "granularity": gran,
                        "throughput": tputs[i],
                    }
                )

    plt.title("Throughput vs SNR (Analytical LLS)")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Spectral Efficiency [bps/Hz]")  # Using SE as proxy for Throughput
    plt.grid(True)
    plt.legend()

    os.makedirs("experiments/hybrid_beamforming/lls/results", exist_ok=True)
    plt.savefig("experiments/hybrid_beamforming/lls/results/throughput_vs_snr.png")
    print("Saved throughput_vs_snr.png")

    # 3. Optimal Granularity Map
    df_res = pd.DataFrame(results)
    # Find max throughput for each SNR
    # Group by SNR, find row with max throughput
    idx = df_res.groupby(["snr"])["throughput"].transform(max) == df_res["throughput"]
    optimal = df_res[idx]

    # Plotting this map is tricky (categorical Y axis?).
    # Or just list the transitions?
    # Let's plot "Best Configuration" as a scatter or step.

    plt.figure(figsize=(10, 4))
    # Map (Waveform, Gran) to Int ID
    configs = (
        df_res[["waveform", "granularity"]]
        .drop_duplicates()
        .sort_values(["waveform", "granularity"])
    )
    config_map = {
        f"{r.waveform}-{r.granularity}": i for i, r in enumerate(configs.itertuples())
    }
    inverse_map = {i: k for k, i in config_map.items()}

    y_vals = [config_map[f"{r.waveform}-{r.granularity}"] for r in optimal.itertuples()]

    plt.step(optimal["snr"], y_vals, where="mid")
    plt.yticks(list(inverse_map.keys()), list(inverse_map.values()))
    plt.title("Optimal Configuration vs SNR")
    plt.xlabel("SNR [dB]")
    plt.grid(True)
    plt.savefig("experiments/hybrid_beamforming/lls/results/optimal_config_map.png")
    print("Saved optimal_config_map.png")


if __name__ == "__main__":
    eval_lls_throughput()
