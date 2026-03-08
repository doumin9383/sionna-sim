import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_sls_metrics(csv_path, output_dir):
    """
    SLSシミュレーション結果（CSV）から各種メトリクスをプロットする (Matplotlib版)
    """
    print(f"Plotting metrics from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("CSV file not found.")
        return

    # Load All Data for Convergence if available
    df_all = df
    all_csv_path = csv_path.replace(".csv", "_all.csv")
    if os.path.exists(all_csv_path):
        df_all = pd.read_csv(all_csv_path)

    # For most plots, use only the LAST iteration to avoid biased statistics
    if "LA_Iter_ID" in df.columns:
        max_iter = df["LA_Iter_ID"].max()
        df_last = df[df["LA_Iter_ID"] == max_iter]
    else:
        df_last = df

    # 1. CDF Plots
    def plot_cdf(data, label, filename, xlabel):
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_data, yvals)
        plt.title(f"CDF of {label} (Final Iteration)")
        plt.xlabel(xlabel)
        plt.ylabel("CDF")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    if "Throughput_Mbps" in df_last.columns:
        plot_cdf(
            df_last["Throughput_Mbps"],
            "Throughput",
            "cdf_throughput.png",
            "Throughput [Mbps]",
        )

    if "SINR_dB" in df_last.columns:
        plot_cdf(df_last["SINR_dB"], "SINR", "cdf_sinr.png", "SINR [dB]")

    # 2. Topology Visualization
    if "UE_Pos_X" in df_last.columns and "BS_Pos_X" in df_last.columns:
        plt.figure(figsize=(10, 10))
        # BS Positions (Unique)
        bs_df = df_last[["BS_Pos_X", "BS_Pos_Y", "BS_ID"]].drop_duplicates()
        plt.scatter(
            bs_df["BS_Pos_X"], bs_df["BS_Pos_Y"], marker="^", s=100, c="red", label="BS"
        )

        # UE Positions (Drop 0)
        drop0_df = df_last[df_last["Drop_ID"] == 0]
        plt.scatter(
            drop0_df["UE_Pos_X"],
            drop0_df["UE_Pos_Y"],
            marker="o",
            s=30,
            alpha=0.5,
            label="UE",
        )

        plt.title("Node Topology (Drop 0)")
        plt.xlabel("X Position [m]")
        plt.ylabel("Y Position [m]")
        plt.axis("equal")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "topology_drop0.png"))
        plt.close()

    # 3. Box Plots
    if "MCS_Index" in df_last.columns:
        plt.figure(figsize=(12, 6))
        plt.boxplot([df_last["MCS_Index"]], vert=False)  # Simple single boxplot
        plt.title("MCS Index Distribution (Final Iteration)")
        plt.xlabel("MCS Index")
        plt.savefig(os.path.join(output_dir, "box_mcs.png"))
        plt.close()

    if "Rank" in df_last.columns:
        plt.figure(figsize=(8, 6))
        counts = df_last["Rank"].value_counts().sort_index()
        plt.bar(counts.index, counts.values)
        plt.title("Rank Distribution (Final Iteration)")
        plt.xlabel("Rank")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "hist_rank.png"))
        plt.close()

    # 4. Scatter Plots
    if "SINR_dB" in df_last.columns and "Throughput_Mbps" in df_last.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_last["SINR_dB"], df_last["Throughput_Mbps"], alpha=0.3)
        plt.title("SINR vs Throughput (Final Iteration)")
        plt.xlabel("SINR [dB]")
        plt.ylabel("Throughput [Mbps]")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "scatter_sinr_throughput.png"))
        plt.close()

    # LA Convergence (Use ALL data here)
    if "LA_Iter_ID" in df_all.columns:
        iter_metrics = (
            df_all.groupby("LA_Iter_ID")[["SINR_dB", "Throughput_Mbps"]]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(10, 5))
        plt.plot(iter_metrics["LA_Iter_ID"], iter_metrics["SINR_dB"], marker="o")
        plt.title("Average SINR Convergence over LA Iterations")
        plt.xlabel("LA Iteration")
        plt.ylabel("Average SINR [dB]")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "convergence_sinr.png"))
        plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        plot_sls_metrics(sys.argv[1], sys.argv[2])
