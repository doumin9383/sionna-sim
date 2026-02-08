# 外部レイトレーシングデータ要求仕様書 (External Ray Tracing Data Requirements)

## 1. 概要 (Overview)

本ドキュメントは、Massive MIMO システムレベルシミュレーション (SLS) において使用する、外部レイトレーシングデータの形式および内容に関する要求仕様を定義するものです。

## 2. 納品データ構成 (Deliverables)

以下の2種類のデータセットを、**セクター (Sector) 単位**で作成・出力してください。
データの集約やランキング（Best Server判定など）はシミュレータ側で行うため、出力側での複雑な処理は不要です。

---

### 2.1. 受信電力マップ (Power Map)

各セクターのカバレッジ状況を確認し、シミュレータ側で干渉計算ペアを特定するために使用します。

* **ファイル名規則**: `sector_{SectorID}_power.mat` (例: `sector_001_power.mat`)
* **フォーマット**: MATLAB .mat (v7.3 推奨)
* **必須変数**:

```matlab
% 受信電力マップの変数構成
MeshIDs % [N x 1] (Integer) 有効な受信電力があるメッシュのユニークID
Powers  % [N x 1] (Double)  対応するメッシュにおける受信電力 [dBm]

```

> **注記**: 受信電力が著しく低い（例: -120dBm以下）メッシュは、データ削減のため除外（Sparse化）して構いません。

---

### 2.2. パスパラメータデータ (Path Data)

Sionna (シミュレータ) 上でチャネル応答を再構成するために必要な物理パラメータです。

* **ファイル名規則**: `sector_{SectorID}_paths.mat` (例: `sector_001_paths.mat`)
* **フォーマット**: MATLAB .mat (v7.3 推奨)
* **必須変数**:

```matlab
% パスパラメータの変数構成
MeshIDs % [N x 1] (Integer) Power Mapと整合性の取れたメッシュIDリスト
Paths   % [1 x N] Struct Array (または Cell Array)
        % 各メッシュ点(i)に対応するマルチパスパラメータ：
        % Paths(i).Gain  : [L x 1] (Complex Double) - 複素パスゲイン
        % Paths(i).Delay : [L x 1] (Double)         - 遅延時間 [sec]
        % Paths(i).DoA   : [L x 2] (Double)         - 到来角 (Azimuth, Elevation) [rad]
        % Paths(i).DoD   : [L x 2] (Double)         - 放射角 (Azimuth, Elevation) [rad]

```

> **注記**:  はパス数を示し、メッシュごとに異なっていても問題ありません。

---

### 2.3. カバレッジ定義 (Common Definition)

シミュレータとレイトレーサ間で座標系を共有するための定義ファイルです。

* **ファイル名**: `coverage_definition.mat`
* **必須変数**:

```matlab
% 共通定義の変数構成
MeshPoints % [M x 3] (Double) - 全メッシュの中心座標 (x, y, z)
MeshIDs    % [M x 1] (Integer) - 各座標に対応するID

```

---

## 3. 前提条件 (Assumptions)

* **座標系**: 直交座標系 (XYZ)。単位はメートル **[m]**。
* **アンテナパターン**:
* レイトレーシング計算時に、基地局・端末双方のアンテナ指向性が考慮されているか、あるいは等方性 (Isotropic) であるかを明記すること。
* **推奨**: アンテナパターンはチャネル再構成時に適用するため、レイトレースデータ自体は **「アンテナ利得を含まない（Isotropic）」** 伝搬チャネル応答であることが望ましい。もしアンテナ利得込みの場合はその旨を連絡すること。


* **周波数**: 中心周波数  を明記すること。

## 4. 補足 (Notes)

* データ量が巨大になるため、.mat のバージョンは **v7.3 (HDF5ベース)** を必須とします。
* 1ファイルあたりのサイズが大きすぎる場合は分割しても構いません。その際は別途、命名規則を協議するものとします。
