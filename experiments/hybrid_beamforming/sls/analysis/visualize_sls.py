import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_cdf(df, col, xlabel, output_dir, filename):
    """CDFプロットを生成するヘルパー関数"""
    if col not in df.columns:
        return
    data = df[col].dropna()
    if len(data) == 0:
        return
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_data, yvals)
    plt.title(f"CDF of {xlabel} (Final Iteration)")
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_resource_allocation(df, output_dir):
    """各セルのリソース割り当て状況をヒートマップで可視化する (全スロット)"""
    # 最終反復のみ抽出
    max_iter = df["LA_Iter_ID"].max()
    df_plot = df[df["LA_Iter_ID"] == max_iter].copy()

    # セル、スロット、UEの情報を整理
    for col in ["BS_ID", "Slot_ID", "UT_Sector_ID", "Allocation_Ratio"]:
        if col not in df_plot.columns:
            print(
                f"カラム {col} が見つからないため、リソース割り当て図をスキップします。"
            )
            return

    bs_ids = sorted(df_plot["BS_ID"].unique())
    num_bs = len(bs_ids)
    slots = sorted(df_plot["Slot_ID"].unique())
    num_slots = len(slots)

    if num_slots < 1:
        print("スロット情報がないため、リソース割り当て図をスキップします。")
        return

    # セルごとにプロット (Grid表示)
    cols = min(3, num_bs)
    rows = (num_bs + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    im = None
    for i, bs_id in enumerate(bs_ids):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        cell_data = df_plot[df_plot["BS_ID"] == bs_id]
        ue_sector_ids = sorted(cell_data["UT_Sector_ID"].unique())
        num_ue_sector = len(ue_sector_ids)

        allocation_matrix = np.zeros((num_ue_sector, num_slots))

        for j, slot in enumerate(slots):
            slot_data = cell_data[cell_data["Slot_ID"] == slot]
            for k, ue_sec_id in enumerate(ue_sector_ids):
                val = slot_data[slot_data["UT_Sector_ID"] == ue_sec_id][
                    "Allocation_Ratio"
                ].values
                if len(val) > 0:
                    allocation_matrix[k, j] = val[0]

        im = ax.imshow(
            allocation_matrix,
            aspect="auto",
            origin="lower",
            extent=[slots[0], slots[-1], 0, num_ue_sector],
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Cell {bs_id}")
        ax.set_yticks(np.arange(num_ue_sector) + 0.5)
        ax.set_yticklabels([f"UE{sid}" for sid in ue_sector_ids])

    fig.text(0.5, 0.04, "Slot Index", ha="center")
    fig.text(0.04, 0.5, "UE (Sector ID)", va="center", rotation="vertical")
    fig.suptitle("Resource Allocation Heatmap per Cell")

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Allocation Ratio (RBGs)")

    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])
    path = os.path.join(output_dir, "resource_allocation_heatmap.png")
    plt.savefig(path)
    print(f"リソース割り当て図を保存しました: {path}")
    plt.close()


def plot_sinr_vs_distance(df, output_dir):
    """SINR vs 距離の散布図"""
    if "Distance_m" not in df or "SINR_dB" not in df:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Distance_m"], df["SINR_dB"], alpha=0.5, c="blue")
    plt.xlabel("Distance [m]")
    plt.ylabel("SINR [dB]")
    plt.title("SINR vs Distance")
    plt.grid(True)
    path = os.path.join(output_dir, "scatter_sinr_distance.png")
    plt.savefig(path)
    print(f"SINR-距離図を保存しました: {path}")
    plt.close()


def plot_convergence(df_all, output_dir):
    """反復ごとの収束プロット"""
    if "LA_Iter_ID" not in df_all.columns:
        return
    iter_metrics = (
        df_all.groupby("LA_Iter_ID")[["SINR_dB", "Throughput_Mbps"]]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 6))
    plt.plot(iter_metrics["LA_Iter_ID"], iter_metrics["SINR_dB"], marker="o")
    plt.title("Average SINR Convergence over LA Iterations")
    plt.xlabel("LA Iteration")
    plt.ylabel("Average SINR [dB]")
    plt.grid(True)
    path = os.path.join(output_dir, "convergence_sinr.png")
    plt.savefig(path)
    plt.close()


def plot_topology(df, output_dir):
    """ノードトポロジーの可視化"""
    if "UE_Pos_X" not in df.columns or "BS_Pos_X" not in df.columns:
        return
    plt.figure(figsize=(10, 10))
    bs_df = df[["BS_Pos_X", "BS_Pos_Y", "BS_ID"]].drop_duplicates()
    plt.scatter(
        bs_df["BS_Pos_X"], bs_df["BS_Pos_Y"], marker="^", s=100, c="red", label="BS"
    )
    for _, row in bs_df.iterrows():
        plt.annotate(f"BS{int(row['BS_ID'])}", (row["BS_Pos_X"], row["BS_Pos_Y"]))

    if "Drop_ID" in df.columns:
        drop0_df = df[df["Drop_ID"] == 0]
        plt.scatter(
            drop0_df["UE_Pos_X"],
            drop0_df["UE_Pos_Y"],
            marker="o",
            s=30,
            alpha=0.5,
            label="UE",
        )
    else:
        plt.scatter(
            df["UE_Pos_X"], df["UE_Pos_Y"], marker="o", s=30, alpha=0.5, label="UE"
        )

    plt.title("Node Topology")
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "topology_drop0.png"))
    plt.close()


def plot_sls_metrics(csv_path, output_dir, csv_all_path=None):
    """メインのプロット関数"""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)

    # 1. CDF
    plot_cdf(
        df, "Throughput_Mbps", "Throughput [Mbps]", output_dir, "cdf_throughput.png"
    )
    plot_cdf(df, "SINR_dB", "SINR [dB]", output_dir, "cdf_sinr.png")

    # 2. Scatter
    if "SINR_dB" in df.columns and "Throughput_Mbps" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df["SINR_dB"], df["Throughput_Mbps"], alpha=0.5)
        plt.xlabel("SINR [dB]")
        plt.ylabel("Throughput [Mbps]")
        plt.title("SINR vs Throughput")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "scatter_sinr_throughput.png"))
        plt.close()

    # 3. Distribution
    if "MCS_Index" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.boxplot(df["MCS_Index"].dropna())
        plt.title("MCS Index Distribution")
        plt.savefig(os.path.join(output_dir, "box_mcs.png"))
        plt.close()

    if "Rank" in df.columns:
        plt.figure(figsize=(8, 6))
        df["Rank"].value_counts().sort_index().plot(kind="bar")
        plt.title("Rank Distribution")
        plt.xlabel("Rank")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "hist_rank.png"))
        plt.close()

    # 4. Topology & Distance
    plot_topology(df, output_dir)
    plot_sinr_vs_distance(df, output_dir)

    # 5. All data related plots
    if csv_all_path is None:
        csv_all_path = csv_path.replace(".csv", "_all.csv")

    if os.path.exists(csv_all_path):
        df_all = pd.read_csv(csv_all_path)
        plot_convergence(df_all, output_dir)
        plot_resource_allocation(df_all, output_dir)

    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        # csv_all_path も渡せるように対応
        all_path = sys.argv[3] if len(sys.argv) > 3 else None
        plot_sls_metrics(sys.argv[1], sys.argv[2], all_path)
