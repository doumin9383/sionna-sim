# wsim-rt 利用ガイド

`wsim.rt` (Ray Tracing) モジュールの外部データロード機能についての解説です。

## 概要

`wsim.rt.external` は、事前に計算されたレイトレーシングデータ（HDF5/Zarr）や、Sionna のリアルタイムレイトレーサーを統一的に扱うためのインターフェースを提供します。

## 主なクラス

### 1. MeshBasedLoader
事前に計算されたレイトレーシングデータをロードするためのクラスです。
空間をメッシュ状に区切り、各メッシュポイントにおけるパス情報（遅延、ゲイン、角度）を保持します。

```python
from wsim.rt.external.loaders import MeshBasedLoader
from sionna.rt import Scene

# シーンのロード（必須）
scene = Scene.load("path/to/scene.xml")

# ローダーの初期化
loader = MeshBasedLoader(
    file_path="path/to/dataset.zarr",
    scene=scene,
    use_3d_search=False # 2D(x,y)で探索するか3D(x,y,z)か
)

# 任意の座標に対するパスを取得
# 内部で KDTree を使用して最近傍のメッシュポイントを検索します
ut_loc = [[100, 200, 1.5], [150, 250, 1.5]] # [NumUT, 3]
paths = loader.get_paths(ut_loc)

# paths は sionna.rt.Paths オブジェクト
# cir = paths.cir() などが呼べる
```

### 2. SionnaLiveTracer
実行時にリアルタイムでレイトレーシングを行うためのラッパーです。

```python
from wsim.rt.external.loaders import SionnaLiveTracer

tracer = SionnaLiveTracer(scene)

# 座標を指定してレイトレーシング実行
# シーン内の受信機(Rx)の位置を更新し、compute_paths() を呼びます
paths = tracer.get_paths(ut_loc)
```

## データフォーマット (Zarr/HDF5)

`MeshBasedLoader` は、Volcano等のレイトレーシングシミュレータから出力されたHDF5/Zarr形式のデータを読み込みます。
Sionnaとの互換性を確保するため、以下のデータ構造と仕様を採用しています。

### 1. データ構造とマッピング

| HDF5 キー / 属性 | 次元 | 単位 | Sionna 対応 | 備考 |
| :--- | :--- | :--- | :--- | :--- |
| **path_gains** | `[N_Mesh, N_TX, MaxP, 2, 2]` | 無次元 (複素振幅) | `Paths.a` | 偏波行列。1W送信・0dBi受信基準で正規化済み。 |
| **delay** | `[N_Mesh, N_TX, MaxP]` | [s] | `Paths.tau` | 遅延。 |
| **zenith_at_tx** | `[N_Mesh, N_TX, MaxP]` | [rad] | `Paths.theta_t` | 送信側(BS) 天頂角 (Global座標)。 |
| **azimuth_at_tx** | `[N_Mesh, N_TX, MaxP]` | [rad] | `Paths.phi_t` | 送信側(BS) 方位角 (Global座標)。 |
| **zenith_at_rx** | `[N_Mesh, N_TX, MaxP]` | [rad] | `Paths.theta_r` | 受信側(UE) 天頂角 (Global座標)。 |
| **azimuth_at_rx** | `[N_Mesh, N_TX, MaxP]` | [rad] | `Paths.phi_r` | 受信側(UE) 方位角 (Global座標)。 |
| **pathloss** | `[N_TX, N_Mesh]` | [dB] | `Paths.lsps['pathloss']` | 広域パスロス (Best Server判定用)。 |
| **tx_positions** | `[N_TX, 3]` | [m] (UTM) | `Transmitter.position` | 基地局/セクタの設置座標。 |
| **tx_orientations** | `[N_TX, 3]` | [deg] | `Transmitter.orientation` | `[Yaw, Pitch, Roll]`. Yaw=Offset, Pitch=Tilt. |
| **tx_antenna_gains** | `[N_TX, 1]` | [dBi] | `AntennaArray.gain` | アンテナのピーク利得。 |
| **tx_names** | `[N_TX, len]` | uint8 | - | 基地局識別名 (ASCII/UTF8バイト列)。 |
| **origin_utm** (Attr) | `[1, 3]` | [m] | `Scene.origin` | シミュレーション空間のUTM原点。 |
| **mesh_step_m** (Attr) | scalar | [m] | `Loader.step` | メッシュ解像度。 |
| **mesh_coordinates** | `[N_Mesh, 3]` | [m] | - | メッシュポイントのUTM座標 (またはローカル座標) |

### 2. 重要な設計仕様

#### 1W 送信電力正規化 (Normalization)
`path_gains` の複素振幅は、**送信電力 1W (30dBm)** で正規化されています。
Sionnaでシミュレーションを実行する際は、`tx_power` を **1.0 (W)** に設定することで、意図した受信電力レベルが再現されます。

#### アンテナ利得の分離 (Isotropic Data)
`path_gains` にはアンテナパターンや指向性利得を含んでいません（Isotropicなデータ）。
Sionnaにロード後、`ExternalPaths` で生成されたパスオブジェクトに対して `apply_antenna_pattern()` を呼び出すことで、Sionna上で定義したアンテナ構成を適用します。
`tx_antenna_gains` や `tx_orientations` のメタデータを使用して `Scene` や `Transmitter` を構成することが推奨されます。

#### 送受信の役割スワップ
Volcano (Uplink) のデータをSionna (Downlink) に合わせるため、データ変換時に **BS側の角度情報をTx、UE側の角度情報をRx** としてマッピングし直されています。

#### 角度体系の変換
Volcanoの角度系（北=0, 水平=0）から、Sionnaの角度系（東=0, 天頂=0）への変換は、データ生成時（HDF5作成時）に行われています。
したがって、`StandardAdapter` は値をそのまま読み込みます。

## インテグレーションのヒント

### SLS (System Level Simulation) への組み込み
SLSでは大量のユーザーを扱うため、`Paths` オブジェクト（全パス情報を含む）をそのままループで回すとオーバーヘッドが大きくなります。
`generate_dataset.py` で生成した軽量な辞書データを扱う既存の `HybridChannelInterface` に対しては、将来的に `MeshBasedLoader` を内部で呼び出し、必要なパラメータだけを抽出して渡す「ラッパー」を作成することが推奨されます。

```python
# 将来的なラッパーのイメージ
class WrapperLoader:
    def __init__(self, mesh_loader):
        self.loader = mesh_loader

    def get_channel_params(self, ut_loc):
        paths = self.loader.get_paths(ut_loc)
        # Pathsオブジェクトから必要なテンソルだけをnumpy/tf辞書に変換
        return {
            "delays": paths.tau,
            "powers": paths.a, # Note: paths.a is amplitude, need power
            ...
        }
```
