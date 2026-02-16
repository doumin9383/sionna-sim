# Sionna Simulator Documentation

## 1. はじめに (Introduction)

本プロジェクトは **Sionna (Ray Tracing)** をベースとした、System Level Simulation (SLS) および Link Level Simulation (LLS) の統合シミュレーターです。
物理層の忠実な再現（Ray Tracing, OFDM波形レベルのPAPR評価）と、大規模なシステムレベル評価の両立を目指しています。

- **Quick Start**: [ユーザーガイド: クイックスタート](user_guide.md#2-クイックスタート)

---

## 2. ユーザーガイド (User Guides)

シミュレーターの利用方法に関するガイドです。

| ドキュメント | 対象者 | 内容 |
| :--- | :--- | :--- |
| **[ユーザーガイド](user_guide.md)** | 全員 | 環境構築、基本的なシミュレーション実行方法、Configの解説。初めにこちらを読んでください。 |
| **[wsim-rt 利用ガイド](wsim_rt_guide.md)** | RT利用者 | 外部レイトレーシングデータ（Zarr/HDF5）のロード方法、`MeshBasedLoader` の使い方。 |

---

## 3. 技術仕様 (Specifications)

ツールの仕様やデータフォーマットに関する詳細な定義です。

| ドキュメント | 内容 |
| :--- | :--- |
| **[外部RTデータ仕様書](spec/external_rt_data_spec.md)** | `wsim.rt` で扱う外部データのスキーマ定義 (Zarr/HDF5)、座標系(UTM vs Local)の変換ルール。 |
| **[ハイブリッドBF実験仕様書](../experiments/hybrid_beamforming/sim_spec.md)** | Hybrid Beamforming 実験における PAPR/MPR 評価手法、SLSの計算フロー、数式定義。 |

---

## 4. 実験・ユースケース (Experiments)

特定のテーマに基づいた実験シナリオと実行手順です。

### [Hybrid Beamforming](../experiments/hybrid_beamforming/README.md)
**目的**: 波形（CP-OFDM vs DFT-s-OFDM）とプリコーディング粒度がシステム性能に与える影響の評価。
- **Phase 1 (LLS)**: PAPR特性の評価とMPRテーブルの作成。
- **Phase 2 (SLS)**: MPRを考慮したマルチセルスループット評価。

---

## 5. 開発者向け (Developer Resources)

- **[技術的負債・TODO](tech_debt.md)**: 既知の課題、将来的なリファクタリング計画、未実装機能のリスト。

### ソースコードマップ (Source Code Map)

主要なコンポーネントの役割と配置を示します。

#### System Level Simulation (SLS)
- **Engine (Core)**
    - [`experiments/hybrid_beamforming/sls/simulator.py`](../experiments/hybrid_beamforming/sls/simulator.py): **`SystemSimulator`**
        - SLSのメインループ。以下のステップでドロップごとの通信品質を評価します。
            1. **Analog Beam Selection** (`_select_analog_beams`): 要素チャネルを取得し、RBG粒度で間引いた上でアナログビームを選択。
            2. **Digital Precoder Calculation** (`_compute_digital_weights`): ポートドメインチャネルをSVD分解し、指定粒度（WB/SB/Carrier）でデジタル重みを計算。
            3. **Link Adaptation & SINR** (`_process_sinr_and_la`): 干渉電力を計算し、ターゲットBLERを満たすMCSを選択。
        - *Note: 現在は実験ディレクトリ内に配置されていますが、将来的には `src/wsim/sls/` への移行が計画されています。*
    - [`experiments/hybrid_beamforming/sls/components/beam_management.py`](../experiments/hybrid_beamforming/sls/components/beam_management.py): **`BeamSelector`**
        - **Sub-panel Sweep**: 第1サブパネルのチャネルのみを使用して最適なDFTビームを探索。
        - **Panel Assignment**: ユーザ $k$ に対して パネル $j$ をラウンドロビンで割り当てる ($k = j \pmod{N_{user}}$)。SU-MIMO時は全パネルが1ユーザに向けられる。

- **Channel Interface**
    - [`src/wsim/sls/channel/interface.py`](../src/wsim/sls/channel/interface.py): **`HybridChannelInterface`**
        - Sionnaの物理層モデルとSLSエンジンの橋渡し。シナリオ制御や簡易的な干渉計算も担当します。
    - [`src/wsim/sls/components/channel_adapters.py`](../src/wsim/sls/components/channel_adapters.py): **`MeshToSLSAdapter`**
        - 外部RTデータ (`MeshBasedLoader`) をSLSが扱える辞書形式に変換します。

#### Ray Tracing (RT)
- **Runner**
    - [`src/wsim/rt/runner.py`](../src/wsim/rt/runner.py): **`SionnaRunner`**
        - レイトレーシングの実行管理。Configに基づくシーン構築、パス計算の実行。
- **Loaders**
    - [`src/wsim/rt/external/loaders.py`](../src/wsim/rt/external/loaders.py): **`ExternalLoaderBase`, `MeshBasedLoader`**
        - Zarr/HDF5形式の外部RTデータの読み込み、空間検索（KDTree）。

#### PHY & Common Utilities
- **PHY Logic**
    - [`src/wsim/common/phy/mcs.py`](../src/wsim/common/phy/mcs.py): **`decode_mcs_index`**
        - MCSインデックス $\leftrightarrow$ 変調多値数/符号化率/所要SINR の変換。DFT-s-OFDM用テーブルもサポート。
    - [`src/wsim/common/phy/pusch.py`](../src/wsim/common/phy/pusch.py): **`PUSCHConfig`**
        - Sionnaのクラスを拡張し、TPMIバリデーションなどを強化。
- **Geometry**
    - [`src/wsim/common/geo.py`](../src/wsim/common/geo.py): **`CoordinateSystem`**
        - UTM座標（緯度経度・高度）とシミュレーションローカル座標（XYZ）の相互変換クラス。
