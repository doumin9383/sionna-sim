# LLS/SLS 分離・並行開発 アーキテクチャ設計書

## 1. 概要
本プロジェクトでは、Sionnaを使用したシミュレータを以下の2つの責務に分割し、並行開発を行います。

*   **LLS (Link Level Simulator)**: 物理層の波形生成、プリコーディング、PAPR特性の評価。最終的に `mpr_table.csv` (Maximum Power Reduction) を出力します。また、将来的に BLER vs SINR テーブルを作成する基盤となります。
*   **SLS (System Level Simulator)**: LLSが生成した MPR テーブルを入力とし、電力制御とMCS (Modulation and Coding Scheme) ベースのリンクアダプテーションを行い、システムスループットを評価します。

## 2. ディレクトリ構造

`experiments/sls_end2end_hybrid_beam/` をベースに開発します。

```
experiments/sls_end2end_hybrid_beam/
├── components/          <-- [共通] LLS/SLSで共有、またはSLSが利用するコンポーネント
│   ├── precoder.py      <-- [共通ロジック] SVD計算アルゴリズム等 (LLS/SLS双方がimportして利用)
│   ├── pusch_model.py   <-- [LLS主体] PUSCHCommunicationModel (Sionnaラッパー)
│   ├── mpr_model.py     <-- [SLS担当] MPR CSV読み込みモデル
│   ├── power_control.py <-- [SLS担当] 送信電力制御ロジック
│   └── link_adaptation.py <-- [SLS担当] MCS選択・TBS計算ロジック
├── lls_scripts/         <-- [LLS担当] LLS専用スクリプト
│   ├── 5G_NR_PUSCH.ipynb <-- [参照] ユーザー提供の参照実装
│   └── run_papr_sim.py  <-- [LLS担当] PAPR計測実行スクリプト
├── simulator.py         <-- [SLS担当] SLSメインスクリプト
└── mpr_table.csv        <-- [成果物] LLSが出力し、SLSが読み込む
```

## 3. 開発フロー (Jujutsu Workspaces)
Gitのブランチ操作によるコンフリクトを避け、効率的に並行作業を行うため `jj workspace` を使用します。

*   **Root Workspace**: 統合管理
*   **LLS Workspace**: `../sionna-lls` (Agent A 作業領域)
*   **SLS Workspace**: `../sionna-sls` (Agent B 作業領域)

## 4. 共通コンポーネント詳細

### `components/precoder.py`
*   **責務**: プリコーディング行列生成アルゴリズムの提供（ライブラリ）。
*   **利用**:
    *   **LLS**: 波形生成時に、理想的なSVDプリコーディングを適用してPAPR/BLERを評価するために使用。
    *   **SLS**: チャネル推定値から最適なプリコーダを算出し、SINR計算に使用。
    *   *注: LLSとSLSでデータの受け渡しは行わず、アルゴリズム（ロジック）のみを共有する。*

### `components/pusch_model.py` (LLS)
*   **クラス**: `PUSCHCommunicationModel`
*   **責務**: `sionna.phy.nr` モジュール (`PUSCHConfig` 等) をラップし、送受信機、チャネル、信号生成のフローを一元管理する。
*   **特記事項**:
    *   DFTS-OFDM (Transform Precoding) 対応のため、手動でのDFT拡散と電力正規化の実装を含む可能性あり。
    *   将来的なBLERシミュレーションにも対応可能な設計とする。

## 5. LLS 設計 (Waveform & PAPR)
*   **入力**: チャネルモデル、波形設定 (CP-OFDM / DFTS-OFDM)
*   **処理**:
    1.  `PUSCHCommunicationModel` による信号生成
    2.  DFTS-OFDMの場合は手動DFT拡散
    3.  オーバーサンプリング & PAPR計測 (CCDF)
*   **出力**: `mpr_table.csv`

## 6. SLS 設計 (Power Control & Link Adaptation)
*   **入力**: `mpr_table.csv`
*   **処理**:
    1.  経路損失 $PL$ 計算
    2.  MPRルックアップ
    3.  送信電力決定 & 電力制御
    4.  `precoder.py` のロジックを用いたプリコーディング行列決定
    5.  SINR計算 & MCS選択
*   **出力**: スループット、SINRマップ
