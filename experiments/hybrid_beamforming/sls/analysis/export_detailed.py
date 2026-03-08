import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


def export_sls_data(history, output_dir, slot_duration=1e-3, max_la_iterations=5):
    """
    history: シミュレータから返された辞書（Tensorの集合）
        Shape: [num_records, batch, num_bs, num_ut_per_sector]
        where num_records = num_drops * max_la_iterations
    output_dir: CSV出力先のディレクトリ
    slot_duration: スループット計算用のスロット期間 [sec]
    max_la_iterations: LA反復回数
    """
    print(f"詳細データをエクスポート中... 出力先: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # テンソルをnumpyに変換
    def to_np(data):
        if isinstance(data, (tf.Tensor, tf.Variable)):
            return data.numpy()
        return np.array(data)

    # 形状の取得
    sample_key = "num_decoded_bits"
    if sample_key not in history:
        print(f"Warning: {sample_key} not found in history.")
        return

    data_shape = to_np(history[sample_key]).shape
    num_records, num_batch, num_bs, num_ut_sector = data_shape
    total_samples = num_records * num_batch * num_bs * num_ut_sector

    # インデックスの生成
    # records_idx: 0, 1, 2, ...
    records_idx = np.repeat(np.arange(num_records), num_batch * num_bs * num_ut_sector)

    # Drop_ID と LA_Iter_ID の計算
    drop_ids = records_idx // max_la_iterations
    la_iter_ids = records_idx % max_la_iterations

    batch_idx = np.tile(
        np.repeat(np.arange(num_batch), num_bs * num_ut_sector), num_records
    )
    bs_idx = np.tile(
        np.repeat(np.arange(num_bs), num_ut_sector), num_records * num_batch
    )
    ut_sector_idx = np.tile(np.arange(num_ut_sector), num_records * num_batch * num_bs)

    # グローバルUE ID
    ue_global_idx = bs_idx * num_ut_sector + ut_sector_idx

    df = pd.DataFrame(
        {
            "Drop_ID": drop_ids,
            "LA_Iter_ID": la_iter_ids,
            "Slot_ID": drop_ids,  # Snapshot simulation assumes Slot_ID = Drop_ID
            "Batch_ID": batch_idx,
            "BS_ID": bs_idx,
            "UT_Sector_ID": ut_sector_idx,
            "UE_Global_ID": ue_global_idx,
        }
    )

    # 各指標の追加
    # Power related metrics will be converted to dBm later
    metrics_map = {
        "Throughput_Bits": "num_decoded_bits",
        "SINR_Lin": "sinr_eff",
        "MCS_Index": "mcs_index",
        "Rank": "rank",
        "PathLoss_dB": "pathloss_serving_cell",
        "Tx_Power_Watt": "tx_power",
        "Beam_Index": "beam_idx",
        "Interference_Power_Lin": "interference_power",
        "Allocation_Ratio": "allocation_mask",  # Ratio of allocated RBGs
        "P_Cmax_dBm": "p_cmax_dbm",
        "MPR_dB": "mpr_db",
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

    # トポロジー情報の追加 (ut_loc, bs_loc)
    # distance の計算も行う
    if "ut_loc" in history and "bs_loc" in history:
        ut_loc_np = to_np(history["ut_loc"]).reshape(-1, 3)
        bs_loc_np = to_np(history["bs_loc"])
        bs_loc_expanded = np.repeat(
            bs_loc_np[:, :, :, np.newaxis, :], num_ut_sector, axis=3
        ).reshape(-1, 3)

        if len(ut_loc_np) == total_samples and len(bs_loc_expanded) == total_samples:
            df["UE_Pos_X"] = ut_loc_np[:, 0]
            df["UE_Pos_Y"] = ut_loc_np[:, 1]
            df["BS_Pos_X"] = bs_loc_expanded[:, 0]
            df["BS_Pos_Y"] = bs_loc_expanded[:, 1]

            # 3D Distance
            df["Distance_m"] = np.linalg.norm(ut_loc_np - bs_loc_expanded, axis=1)

    # 派生指標の計算
    if "SINR_Lin" in df:
        df["SINR_dB"] = 10 * np.log10(np.maximum(df["SINR_Lin"], 1e-20))

    if "Interference_Power_Lin" in df:
        df["Interference_Power_dBm"] = (
            10 * np.log10(np.maximum(df["Interference_Power_Lin"], 1e-20)) + 30
        )

    if "Tx_Power_Watt" in df:
        df["Tx_Power_dBm"] = 10 * np.log10(np.maximum(df["Tx_Power_Watt"], 1e-20)) + 30

    if "Throughput_Bits" in df:
        # Mbps単位に変換: (Bits / Duration) / 1e6
        df["Throughput_Mbps"] = (df["Throughput_Bits"] / slot_duration) / 1e6

    # CSV保存 (途中経過を含む全データ)
    csv_path = os.path.join(output_dir, "detailed_results_all.csv")
    df.to_csv(csv_path, index=False)
    print(f"詳細CSV（全反復）を保存しました: {csv_path}")

    # 最終反復のみのデータを保存
    df_last = df[df["LA_Iter_ID"] == (max_la_iterations - 1)].copy()
    csv_path_last = os.path.join(output_dir, "detailed_results.csv")
    df_last.to_csv(csv_path_last, index=False)
    print(f"詳細CSV（最終反復のみ）を保存しました: {csv_path_last}")

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
