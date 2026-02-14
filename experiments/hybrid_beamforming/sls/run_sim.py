import os
import sys
import tensorflow as tf

# --- Add project root to path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
# sys.path.append(project_root)

# Import Local Components
from experiments.hybrid_beamforming.sls.simulator import (
    SystemSimulator,
)
from experiments.hybrid_beamforming.sls.configs import SLSConfig
import csv


def run_test():
    # Master Config
    # デフォルトのプロダクション設定を使用（batch_size=32, num_ut_per_sector=4, num_slots=20）
    # 必要に応じてここでオーバーライド可能
    config = SLSConfig()

    # 4. Instantiate Simulator
    # ResourceGridやPanelArrayはSimulator内部でConfigから生成される
    sim = SystemSimulator(config=config)

    # 5. Run Simulation
    print(
        f"シミュレーションを開始します... (Batch Size: {config.batch_size}, Drops: {config.num_ut_drops}, UTs/Sector: {config.num_ut_per_sector})"
    )

    # Enable XLA for potential speedup if available, but for debugging eager might be safer
    # tf.config.optimizer.set_jit(True)

    # Run
    # Returns a dictionary of Tensors
    # configのtx_powerを使用
    history = sim(config.num_ut_drops, config.bs_max_power_dbm)

    print("シミュレーション完了。")
    print("History keys:", history.keys())

    # Save results to a pickle file for comprehensive analysis
    import pickle

    os.makedirs(config.output_dir, exist_ok=True)
    history_path = os.path.join(config.output_dir, "history.pkl")

    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"履歴データを保存しました: {history_path}")

    # Calculate Average Throughput for quick check
    # num_decoded_bits: [slots, batch, bs, ut_per_sector]
    if "num_decoded_bits" in history:
        bits = history["num_decoded_bits"]
        # bps単位のスループット合計
        total_throughput_bps = tf.reduce_sum(bits, axis=[1, 2, 3])
        avg_throughput_mbps = tf.reduce_mean(total_throughput_bps) / 1e6
        print(f"平均ネットワークスループット: {avg_throughput_mbps:.2f} Mbps")

    # Simple CSV export for legacy plotting compatibility
    try:
        csv_path = os.path.join(config.output_dir, "simulation_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Slot", "Average_Throughput_bps"])
            # Writing total network throughput per slot
            tput_vals = total_throughput_bps.numpy()
            for i, val in enumerate(tput_vals):
                writer.writerow([i, val])
        print(f"サマリーCSVを保存しました: {csv_path}")
    except Exception as e:
        print(f"サマリーCSVの保存に失敗しました: {e}")


if __name__ == "__main__":
    run_test()
