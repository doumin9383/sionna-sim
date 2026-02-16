import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_sls_metrics(csv_path, output_dir):
    """
    csv_path: detailed_results.csv へのパス
    output_dir: 図の保存先
    """
    print(f"可視化を実行中... 保存先: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # 1. Throughput & SINR CDF
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if "Throughput_Mbps" in df:
        sorted_data = np.sort(df["Throughput_Mbps"])
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        plt.plot(sorted_data, yvals)
        plt.title("Throughput CDF")
        plt.xlabel("Throughput [Mbps]")
        plt.ylabel("CDF")
        plt.grid(True)

    plt.subplot(1, 2, 2)
    if "SINR_dB" in df:
        sorted_data = np.sort(df["SINR_dB"])
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        plt.plot(sorted_data, yvals)
        plt.axvline(x=0, color="r", linestyle="--", label="0dB")
        plt.title("SINR CDF")
        plt.xlabel("SINR [dB]")
        plt.ylabel("CDF")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cdf_metrics.png"))
    plt.close()

    # 2. MCS & Beam Distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if "MCS_Index" in df:
        df["MCS_Index"].hist(bins=range(30), rwidth=0.8, align="left")
        plt.title("MCS Index Distribution")
        plt.xlabel("MCS Index")
        plt.ylabel("Count")
        plt.grid(True)

    plt.subplot(1, 2, 2)
    if "Beam_Index" in df:
        df["Beam_Index"].value_counts().sort_index().plot(kind="bar")
        plt.title("Beam Usage Distribution")
        plt.xlabel("Beam Index")
        plt.ylabel("Count")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_metrics.png"))
    plt.close()

    # 3. Simple Analysis Report
    report_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== SLS Simulation Analysis Summary ===\n")
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Avg Throughput: {df['Throughput_Mbps'].mean():.4f} Mbps\n")
        if "SINR_dB" in df:
            f.write(f"Avg SINR: {df['SINR_dB'].mean():.2f} dB\n")
            f.write(f"5th percentile SINR: {df['SINR_dB'].quantile(0.05):.2f} dB\n")

        f.write("\n--- Potential Issues ---\n")
        if df["Throughput_Mbps"].mean() < 1.0:
            f.write("[CAUTION] Throughput is very low (< 1 Mbps).\n")
            if df["SINR_dB"].mean() < 5:
                f.write(" -> Reason: SINR seems to be the bottleneck (Avg < 5dB).\n")
                if "Interference_Power_dBm" in df:
                    avg_i = df["Interference_Power_dBm"].mean()
                    f.write(f" -> Avg Interference Power: {avg_i:.2f} dBm\n")
            elif df["MCS_Index"].mean() < 2:
                f.write(" -> Reason: MCS is stuck at 0 or 1 despite OK SINR.\n")

        if "Beam_Index" in df:
            unique_beams = df["Beam_Index"].nunique()
            f.write(f"\nUnique Beams used: {unique_beams}\n")
            if unique_beams == 1:
                f.write(
                    "[WARNING] Only 1 beam type is used. Beam management might be stuck.\n"
                )

    print(f"簡易解析レポートを保存しました: {report_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        output_dir = os.path.dirname(csv_path)
        plot_sls_metrics(csv_path, output_dir)
