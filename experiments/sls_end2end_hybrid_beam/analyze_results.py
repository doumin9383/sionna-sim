import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# 結果ファイルのパス
history_path = "/home/sh-fukue/Documents/Developments/sionna-sim/experiments/sls_end2end_hybrid_beam/results/test_run_01_system/history.pkl"

if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    print(f"Keys in history: {history.keys()}")

    # 統計情報の表示
    # num_decoded_bits: [num_slots, batch_size, num_bs, num_ut_per_sector]
    bits = history["num_decoded_bits"]
    throughput_per_slot = np.sum(bits, axis=(1, 2, 3))

    print("\n--- Simulation Results Summary ---")
    print(f"Simulated Slots: {bits.shape[0]}")
    print(f"Total Bits Decoded: {np.sum(bits)}")
    print(f"Throughput per slot (total): {throughput_per_slot}")

    # SINRの統計
    sinr_eff = history["sinr_eff"]
    print(
        f"Average effective SINR [dB]: {10 * np.log10(np.mean(sinr_eff)) if np.mean(sinr_eff) > 0 else -np.inf}"
    )

    # 簡単な可視化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(throughput_per_slot)
    plt.title("Total Throughput per Slot")
    plt.xlabel("Slot Index")
    plt.ylabel("Bits")

    plt.subplot(1, 2, 2)
    # Convert to dB for plotting
    sinr_db = 10 * np.log10(np.maximum(sinr_eff.numpy(), 1e-10))
    plt.hist(sinr_db.flatten(), bins=20)
    plt.title("SINR Distribution")
    plt.xlabel("Effective SINR [dB]")

    plt.tight_layout()
    plt.savefig(
        "/home/sh-fukue/Documents/Developments/sionna-sim/experiments/sls_end2end_hybrid_beam/results/summary_plot.png"
    )
    print("\nSummary plot saved to results/summary_plot.png")
else:
    print(f"Error: {history_path} not found.")
