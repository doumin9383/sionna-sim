# 外部レイトレーシングデータサブシステム技術仕様書

## 1. 概要 (Overview)

本システムは、事前計算されたレイトレーシング結果（Zarr/HDF5フォーマット）のロード、または Sionna のリアルタイムレイトレーサー（Live RT）の実行を、統一されたインターフェースで制御するためのサブシステムです。

### 主要コンポーネント

- **`ExternalLoaderBase`**: 抽象基底クラス。`get_paths(ut_coords_local)` メソッドを定義し、ローダーの共通インターフェースを提供します。
- **`MeshBasedLoader`**: 事前計算されたメッシュ状のデータ点から最近傍探索を用いてパスデータを取得します。
- **`SionnaLiveTracer`**: Sionna の `Scene.compute_paths` をラップし、リアルタイムでのレイトレーシング結果を返します。
- **`ExternalPaths`**: 外部データを Sionna の `Paths` オブジェクトとして扱えるように拡張されたクラス。

## 2. 座標系 (Coordinate System)

シミュレーターは、グローバルな UTM 座標とシミュレーション内のローカル座標を使い分けます。変換の基準となるアンカーポイントとして `origin_utm` を使用します。

### 変換式

ローカル座標系 $(x, y, z)$ と UTM 座標系 $(E, N, A)$ の関係は以下の通りです：

$$ \text{Local} = \text{UTM} - \text{Origin} $$
$$ \text{UTM} = \text{Local} + \text{Origin} $$

- **Origin**: ローカル原点 $(0, 0, 0)$ に対応する UTM 座標 $(E_0, N_0, A_0)$。

## 3. データスキーマ (Data Schema)

Zarr または HDF5 ファイルは以下の構造を持つ必要があります。

### 属性 (Attributes)

| 名前 | 説明 | 型 |
| :--- | :--- | :--- |
| `origin_utm` | ローカル原点の UTM 座標 (Easting, Northing, Altitude) | `float[3]` |
| `num_tx` | 送信点 (Base Station) の数 | `int` |

### データセット (Datasets)

| 名前 | 形状 (Shape) | 説明 | 型 |
| :--- | :--- | :--- | :--- |
| `mesh_coordinates` | `[N_Mesh, 3]` | メッシュ点の UTM 座標 | `float32` |
| `path_gains` | `[N_Mesh, Num_TX, Max_Paths, 2, 2]` | 複素偏波行列 (Polarization Transfer Matrix) | `complex64` |
| `path_gain` | `[N_Mesh, Num_TX, Max_Paths]` | スカラー利得（`path_gains` がない場合のフォールバック） | `float32` |
| `phase` | `[N_Mesh, Num_TX, Max_Paths]` | 位相（`path_gains` がない場合のフォールバック） | `float32` |
| `delay` | `[N_Mesh, Num_TX, Max_Paths]` | 伝搬遅延 [s] | `float32` |
| `zenith_at_tx` | `[N_Mesh, Num_TX, Max_Paths]` | 送信点での天頂角 [rad] | `float32` |
| `azimuth_at_tx` | `[N_Mesh, Num_TX, Max_Paths]` | 送信点での方位角 [rad] | `float32` |
| `zenith_at_rx` | `[N_Mesh, Num_TX, Max_Paths]` | 受信点での天頂角 [rad] | `float32` |
| `azimuth_at_rx` | `[N_Mesh, Num_TX, Max_Paths]` | 受信点での方位角 [rad] | `float32` |

## 4. 運用上の重要なルール (Important Rules)

### 全方向性放射の前提 (Omni-directional Assumption)

外部データとして保存されるパスデータは、**送信・受信アンテナ共に完全無指向性（Isotropic）であると仮定**して計算されている必要があります。
実際のアンテナパターン（利得および偏波特性）は、シミュレーターの実行時に Sionna のアンテナモデル（`PlanarArray` 等）を介して適用されます。

### 正規化 (Normalization)

`path_gains` または `path_gain` に記録される値は、**送信電力を 1W [30dBm] と仮定**した際の受信複素振幅または受信電力である必要があります。
実際の送信電力制御は、シミュレーター内のパワー制御ロジックによって実行時に適用されます。
