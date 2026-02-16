import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


def export_sls_data(history, output_dir, slot_duration=1e-3):
    """
    history: シミュレータから返された辞書（Tensorの集合）
    output_dir: CSV出力先のディレクトリ
    slot_duration: スループット計算用のスロット期間 [sec]
    """
    print(f"詳細データをエクスポート中... 出力先: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # テンソルをnumpyに変換
    def to_np(data):
        if isinstance(data, (tf.Tensor, tf.Variable)):
            return data.numpy()
        return np.array(data)

    # 形状の取得 [Slots, Batch, BS, UT_per_sector]
    # 例: [20, 32, 3, 4]
    sample_key = "num_decoded_bits"
    if sample_key not in history:
        print(f"Warning: {sample_key} not found in history.")
        return

    data_shape = to_np(history[sample_key]).shape
    num_slots, num_batch, num_bs, num_ut_sector = data_shape
    total_samples = num_slots * num_batch * num_bs * num_ut_sector

    # インデックスの生成
    slots_idx = np.repeat(np.arange(num_slots), num_batch * num_bs * num_ut_sector)
    batch_idx = np.tile(
        np.repeat(np.arange(num_batch), num_bs * num_ut_sector), num_slots
    )
    bs_idx = np.tile(np.repeat(np.arange(num_bs), num_ut_sector), num_slots * num_batch)
    ut_sector_idx = np.tile(np.arange(num_ut_sector), num_slots * num_batch * num_bs)

    # グローバルUE ID
    ue_global_idx = bs_idx * num_ut_sector + ut_sector_idx

    df = pd.DataFrame(
        {
            "Drop_ID": slots_idx,
            "Batch_ID": batch_idx,
            "BS_ID": bs_idx,
            "UE_Sector_ID": ut_sector_idx,
            "UE_Global_ID": ue_global_idx,
        }
    )

    # 各指標の追加
    metrics_map = {
        "Throughput_Bits": "num_decoded_bits",
        "SINR_Lin": "sinr_eff",
        "MCS_Index": "mcs_index",
        "Rank": "rank",
        "PathLoss_dB": "pathloss_serving_cell",
        "Tx_Power_Watt": "tx_power",
        "Beam_Index": "beam_idx",
        "Interference_Power_Lin": "interference_power",
    }

    for col_name, hist_key in metrics_map.items():
        if hist_key in history:
            val = to_np(history[hist_key]).flatten()
            if len(val) == total_samples:
                df[col_name] = val
            else:
                print(
                    f"Skipping {col_name}: shape mismatch {len(val)} vs {total_samples}"
                )

    # 派生指標の計算
    if "SINR_Lin" in df:
        df["SINR_dB"] = 10 * np.log10(np.maximum(df["SINR_Lin"], 1e-20))

    if "Interference_Power_Lin" in df:
        df["Interference_Power_dBm"] = (
            10 * np.log10(np.maximum(df["Interference_Power_Lin"], 1e-20)) + 30
        )

    if "Throughput_Bits" in df:
        # Mbps単位に変換: (Bits / Duration) / 1e6
        df["Throughput_Mbps"] = (df["Throughput_Bits"] / slot_duration) / 1e6

    # CSV保存
    csv_path = os.path.join(output_dir, "detailed_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"詳細CSVを保存しました: {csv_path}")
    return df


if __name__ == "__main__":
    # 単体テスト用
    import sys

    if len(sys.argv) > 1:
        history_path = sys.argv[1]
        output_dir = os.path.dirname(history_path)
        with open(history_path, "rb") as f:
            hist = pickle.load(f)
        export_sls_data(hist, output_dir)
