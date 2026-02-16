import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf


def analyze_sls_results(history_path, output_dir):
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found.")
        return

    print(f"Loading history from {history_path}...")
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    # Helper to convert Tensor to Numpy (if stored as Tensor)
    def to_np(data):
        if isinstance(data, (tf.Tensor, tf.Variable)):
            return data.numpy()
        return np.array(data)

    # 1. Throughput Analysis
    # num_decoded_bits shape: [slots, batch, bs, ut_per_sector]
    if "num_decoded_bits" in history:
        bits = to_np(history["num_decoded_bits"])
        # Throughput (Mbps) per user per slot
        # Assuming slot duration 1ms (defualt) or need to get it.
        # Let's assume 1 slot = 1ms for now or just plot bits per slot.
        # Ideally we know slot duration.
        tput_mbps = bits / 1e6 * 1000.0  # Approx if 1ms

        # Flatten for CDF
        tput_flat = tput_mbps.flatten()

        # CDF Plot
        sorted_tput = np.sort(tput_flat)
        cdf = np.arange(len(sorted_tput)) / float(len(sorted_tput))

        plt.figure(figsize=(8, 6))
        plt.plot(sorted_tput, cdf)
        plt.title("Throughput CDF (System Level)")
        plt.xlabel("Throughput [Mbps]")
        plt.ylabel("CDF")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "throughput_cdf.png"))
        plt.close()
        print("Saved throughput_cdf.png")

        # Time Series (Network Sum)
        network_sum_tput = np.sum(tput_mbps, axis=(1, 2, 3))
        plt.figure(figsize=(10, 5))
        plt.plot(network_sum_tput)
        plt.title("Total Network Throughput Time Series")
        plt.xlabel("Slot")
        plt.ylabel("Throughput [Mbps]")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "network_throughput_time_series.png"))
        plt.close()

    # 2. Adaptation Statistics (MCS)
    if "mcs_index" in history:
        mcs = to_np(history["mcs_index"])
        mcs_flat = mcs.flatten()

        plt.figure(figsize=(8, 6))
        plt.hist(mcs_flat, bins=range(30), align="left", rwidth=0.8)
        plt.title("MCS Index Distribution")
        plt.xlabel("MCS Index")
        plt.ylabel("Count")
        plt.grid(True, axis="y")
        plt.savefig(os.path.join(output_dir, "mcs_distribution.png"))
        plt.close()
        print("Saved mcs_distribution.png")

    # 3. SINR Analysis
    if "sinr_eff" in history:
        sinr = to_np(history["sinr_eff"])
        # Convert to dB
        sinr_db = 10 * np.log10(np.maximum(sinr, 1e-20))
        sinr_flat = sinr_db.flatten()

        # CDF
        sorted_sinr = np.sort(sinr_flat)
        cdf_sinr = np.arange(len(sorted_sinr)) / float(len(sorted_sinr))

        plt.figure(figsize=(8, 6))
        plt.plot(sorted_sinr, cdf_sinr)
        plt.title("Effective SINR CDF")
        plt.xlabel("SINR [dB]")
        plt.ylabel("CDF")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "sinr_cdf.png"))
        print("Saved sinr_cdf.png")


if __name__ == "__main__":
    # Default paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "results")
    history_path = os.path.join(output_dir, "history.pkl")

    analyze_sls_results(history_path, output_dir)
