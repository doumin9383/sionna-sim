# SLS Integration Troubleshooting Report (2026-02-18)

本ドキュメントでは、SLS (System Level Simulation) と外部RTデータ (`ExternalLoader`) の統合時に発生した一連の次元エラー、型エラー、およびロジックエラーの詳細と、その修正内容を記録します。

## 1. 概要
`verify_sls_integration.py` を用いた統合テストにおいて、TensorFlowの `InvalidArgumentError` (Dimension mismatch) や `AttributeError` が多発し、シミュレーションが完了しない状態でした。以下の4つの主要な問題を特定し、修正を行いました。

1.  **テンソルランクの不整合 (Rank 8 Error)**
2.  **`get_power_map` メソッドの欠落**
3.  **ダミーデータ (`pathloss`) の形状不整合**
4.  **BS側重み配列の次元定義ミス (Broadcasting Error)**

---

## 2. 詳細と修正内容

### 2.1 テンソルランクの不整合 (Rank 8 Issue)

#### 問題
`ExternalLoader` (DrJitベース) から取得したチャネルデータ (`h_srv_port`, `h_batch_port`) が、想定される **Rank 7** (`[Batch, UT, Neighbor, RxP, TxP, Time, Freq]`) ではなく、**Rank 8** (`[..., RxP, TxP, Extra, Time, Freq]`) として返されるケースが確認されました。
これにより、後続の `tf.transpose` や `reshape` 操作で次元数が合わず、エラーが発生していました。

#### 原因
DrJit/Mitsubaのバージョンや設定、あるいは `ExternalPaths` の読み込み処理において、特定の条件下で余分な次元（おそらく偏波やマルチサンプル次元）が含まれてしまうためと考えられます。

#### 修正 (`src/wsim/sls/simulator.py`)
`_compute_digital_weights` および `_compute_interference` メソッドにおいて、入力テンソルのランクを動的に判定し、スライス処理を分岐させることで堅牢性を確保しました。

```python
# Before
h_srv_sliced = h_srv_port[:, :, 0, :, :, 0, :] # Fixed slicing for Rank 7

# After (Robust)
if len(h_srv_port.shape) == 8:
    # Remove Neighbor(2), Extra(5), Time(6)
    # Sliced: [B, U, RxP, TxP, F]
    h_srv_sliced = h_srv_port[:, :, 0, :, :, 0, 0, :]
else:
    # Standard Rank 7: [B, U, N, RxP, TxP, T, F]
    # Remove Neighbor(2), Time(5)
    h_srv_sliced = h_srv_port[:, :, 0, :, :, 0, :]
```

### 2.2 `get_power_map` メソッドの欠落

#### 問題
シミュレータの電力制御ロジック (`_apply_power_control`) は、チャネルモデルが `get_power_map(ut_indices)` を実装していることを期待していましたが、`SLSExternalLoader` にはこのメソッドが存在せず、`AttributeError` が発生しました。

#### 修正 (`experiments/hybrid_beamforming/sls/external_loader.py`)
`SLSExternalLoader` クラスに `get_power_map` メソッドを実装しました。
外部データに含まれるパスごとの電力（Path Power）を合計し、対数変換して受信電力（RSRP, dBm）として返します。

```python
    def get_power_map(self, ut_indices):
        # ... (get_rays) ...
        powers_linear = ray_data["powers"] # [Batch, TX, RX, Paths]
        # Sum over paths -> Total Gain
        channel_gain_linear = tf.reduce_sum(powers_linear, axis=-1)
        # Convert to dB and add Tx Power
        # ...
        return powers_dbm
```

### 2.3 ダミーデータ (`pathloss`) の形状不整合

#### 問題
テスト用データ生成スクリプト `create_dummy_hdf5.py` が生成する `pathloss` データの形状が `[Num_TX, Num_Mesh]` (TX-major) でした。
一方、`ExternalPaths` クラスはデータを読み込む際、Mesh（受信点）インデックスでスライスすることを想定しており、`[Num_Mesh, Num_TX]` (RX-major) の形状を期待していました。
この不整合により、Zarr/HDF5読み込み時にインデックスが範囲外となり、`IndexError` (Bounds Check Error) が発生しました。

#### 修正 (`src/wsim/rt/create_dummy_hdf5.py`)
生成時の形状を `(num_mesh, num_tx)` に修正しました。

```python
- pathloss = np.random.uniform(..., size=(num_tx, num_mesh))
+ pathloss = np.random.uniform(..., size=(num_mesh, num_tx))
```

### 2.4 BS側重み配列の次元定義ミス (Broadcasting Error)

#### 問題
シミュレータ内のBS側デジタル重みバッファ `w_bs_dig_full` が `[Batch, N_BS, ...]` で定義されていました。
しかし、ランク最適化処理 (`_optimize_rank_allocation`) や重み適用処理 (`_get_effective_weights`) では、UTごとのマスク処理を行うため、`N_UT` 次元のテンソルとの演算が発生します。
`N_BS != N_UT`（例: BS=3, UT=6）の場合、形状が一致せずブロードキャストエラーが発生しました。
また、同一BS配下に複数のUTが存在する場合、BSインデックスベースで重みを保存すると上書きが発生し、正しく動作しませんでした。

#### 修正 (`src/wsim/sls/simulator.py`)
`w_bs_dig_full` の次元を **UT単位** (`N_UT_Total`) に変更しました。これにより、各UTにとっての「Serving BS側の重み」を個別に保持できるようになり、エラーが解消されました。

```python
# Before
w_bs_dig_full = tf.zeros([B, N_BS, ...])

# After
w_bs_dig_full = tf.zeros([B, N_UT_Total, ...])
# Updated using indices_ut_flat instead of indices_bs_flat
```

---

## 3. 検証
修正後、統合テストスクリプト `verify_sls_integration.py` を実行し、以下の点を確認しました。

*   シミュレーションがエラーなく完走する (Exit Code 0)。
*   `num_decoded_bits` 等のPHY層メトリクスが正しく記録されている。
*   Rank 8 および Rank 7 の両方のケースで動作する（ロバスト性の確保）。
