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

`MeshBasedLoader` が期待するデータ構造は以下の通りです。
`src/wsim/rt/external/ingester.py` を使用して変換・生成することが推奨されます。

| Key | Shape | 説明 |
| :--- | :--- | :--- |
| `mesh_coordinates` | `[NumPoints, 3]` | メッシュポイントのUTM座標 (またはローカル座標) |
| `path_gains` | `[NumPoints, NumTx, NumPaths, 2, 2]` | 偏波込みの複素パスゲイン |
| `delay` | `[NumPoints, NumTx, NumPaths]` | 遅延時間 [s] |
| `zenith_at_rx` | `[NumPoints, NumTx, NumPaths]` | 受信天頂角 [rad] |
| `azimuth_at_rx` | `[NumPoints, NumTx, NumPaths]` | 受信方位角 [rad] |
| ... | ... | (Tx側の角度も同様) |

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
