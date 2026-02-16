import os
import sys
import tensorflow as tf

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
        # history["num_decoded_bits"] is scaled by num_data_symbols in simulator
        # We need the physical duration of the slot (including DMRS/overhead symbols)
        # assuming sim.slot_duration is the duration of ONE OFDM symbol (including CP)
        slot_time_duration = sim.slot_duration * config.num_symbols_per_slot

        # Mbps単位のスループット計算: (全UEの合計ビット数 / スロット期間[s]) / 1e6
        total_bits_per_slot = tf.reduce_sum(bits, axis=[1, 2, 3])
        avg_throughput_mbps = (
            tf.reduce_mean(total_bits_per_slot / slot_time_duration) / 1e6
        )
        print(f"平均ネットワークスループット: {avg_throughput_mbps:.2f} Mbps")

    # Advanced Analysis and Visualization
    print("\n詳細解析を実行します...")
    from experiments.hybrid_beamforming.sls.analysis.export_detailed import (
        export_sls_data,
    )
    from experiments.hybrid_beamforming.sls.analysis.visualize_sls import (
        plot_sls_metrics,
    )

    # CSV出力と可視化
    export_sls_data(
        history,
        config.output_dir,
        slot_duration=slot_time_duration,
        max_la_iterations=config.max_la_iterations,
    )

    # パラメータの保存
    export_config_to_csv(config, config.output_dir)

    # 詳細ログ（総当り結果）の保存
    if hasattr(sim, "detailed_logs") and sim.detailed_logs:
        import pandas as pd

        detailed_log_path = os.path.join(
            config.output_dir, "detailed_rank_selection.csv"
        )
        df_detailed = pd.DataFrame(sim.detailed_logs)
        df_detailed.to_csv(detailed_log_path, index=False)
        print(f"詳細なランク選択ログを保存しました: {detailed_log_path}")

    plot_sls_metrics(
        os.path.join(config.output_dir, "detailed_results.csv"), config.output_dir
    )

    print(f"\nすべての結果が {config.output_dir} に保存されました。")


def export_config_to_csv(config, output_dir):
    """
    設定パラメータをCSVに保存する
    """
    import csv
    from dataclasses import asdict

    csv_path = os.path.join(output_dir, "simulation_parameters.csv")
    try:
        # dataclassから辞書へ変換（再帰的ではないが、Configはフラットに近い）
        # SimulationCommonConfigの継承分も含めるため asdict が便利
        config_dict = asdict(config)

        # 不要なオブジェクト（PanelArray等）は除外または文字列表現にする
        # PanelArrayはシリアライズできない可能性が高いので、文字列表現にするか除外
        filtered_dict = {}
        for k, v in config_dict.items():
            if k in ["bs_array", "ut_array"]:
                filtered_dict[k] = str(v)
            else:
                filtered_dict[k] = v

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value"])
            for key, value in filtered_dict.items():
                writer.writerow([key, value])
        print(f"パラメータCSVを保存しました: {csv_path}")

    except Exception as e:
        print(f"Warning: Failed to export config to CSV: {e}")


if __name__ == "__main__":
    run_test()
