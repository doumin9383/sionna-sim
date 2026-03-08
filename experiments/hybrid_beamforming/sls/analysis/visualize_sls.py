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
    """各セルのリソース割り当て状況をヒートマップで可視化する (全スロット)
    縦軸: サブキャリア(RB)
    横軸: スロット(Drop)
    色   : UE-ID
    """
    npy_path = os.path.join(output_dir, "bs_alloc_map.npy")
    if not os.path.exists(npy_path):
        print(
            f"RBベース割り当てマップが見つからないため、リソース割り当て図をスキップします。({npy_path})"
        )
        return

    # csvには最終LA結果が含まれるとしているが、NPYは全レコードある。
    max_iter = df["LA_Iter_ID"].max()
    bs_alloc_map = np.load(npy_path)  # shape: [num_records, batch, num_bs, num_rb]
    R, B, NBS, NRB = bs_alloc_map.shape

    num_la_iter = int(max_iter) + 1
    num_drops = R // num_la_iter
    if num_drops * num_la_iter != R:
        print("Warning: NPY shape mismatch with expected records")
        return

    # [num_drops, num_la_iter, batch, num_bs, num_rb] に変形し、最終反復を抽出
    bs_alloc_map_reshaped = bs_alloc_map.reshape(num_drops, num_la_iter, B, NBS, NRB)
    final_alloc = bs_alloc_map_reshaped[
        :, max_iter, 0, :, :
    ]  # assuming batch=0. shape: [num_drops, num_bs, num_rb]

    num_slots = num_drops
    num_bs = NBS
    num_rb = NRB

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
    import matplotlib.colors as mcolors

    # -1(未割当)とUE-ID(0,1,2...)を色分けするためのColormapを作成
    cmap = plt.cm.get_cmap("tab10", 10).copy()
    cmap.set_under("white")  # -1 は白

    for i in range(num_bs):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        # 横: Slot(num_drops), 縦: RB
        allocation_matrix = final_alloc[:, i, :].T  # [num_rb, num_slots]

        im = ax.imshow(
            allocation_matrix,
            aspect="auto",
            origin="lower",
            extent=[-0.5, num_slots - 0.5, -0.5, num_rb - 0.5],
            cmap=cmap,
            vmin=-0.5,
            vmax=9.5,
            interpolation="nearest",
        )
        ax.set_title(f"Cell {i}")
        if c == 0:
            ax.set_ylabel("Subcarrier (RB)")
        if r == rows - 1:
            ax.set_xlabel("Slot Index")
        # 縦軸の目盛りを少し間引く
        ax.set_yticks(np.arange(0, num_rb, max(1, num_rb // 10)))

    fig.suptitle("Resource Allocation Heatmap per Cell")

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(
            im, cax=cbar_ax, label="UE ID (White: Unallocated)", ticks=np.arange(0, 10)
        )

    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])
    path = os.path.join(output_dir, "resource_allocation_heatmap.png")
    plt.savefig(path)
    print(f"RBベース割り当て図を保存しました: {path}")
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
