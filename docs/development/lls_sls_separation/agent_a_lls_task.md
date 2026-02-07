# Agent A (LLS担当) 作業指示書

あなたは **LLS (Link Level Simulator)** の実装担当です。
`../sionna-lls` ワークスペースで作業を行ってください。

## 目的
Sionna のチュートリアル `5G_NR_PUSCH.ipynb` の内容をベースに、将来的な拡張（BLER-SIRテーブル作成等）を見据えた `PUSCHCommunicationModel` クラスを実装し、それを用いて PAPR (Peak-to-Average Power Ratio) を評価して MPR (Maximum Power Reduction) テーブルを作成することです。

**重要**:
*   自己流の実装は避け、Sionnaの標準API (`PUSCHConfig`, `PUSCHTransmitter`) を最大限活用してください。
*   本シミュレータの設定方針（Dataclassベース等）に従ってください。
*   将来的にBLER-SIRテーブルが必要になるため、チュートリアルの実装（Transmitter, Channel, Receiver, BER計測まで）は基本的に網羅してください。

## タスク詳細

### 1. `components/precoder.py` の実装
*   **役割**: プリコーディング行列生成ロジックの提供 (ライブラリ的な位置づけ)
*   **仕様**:
    *   **SLSとLLSで共通のアルゴリズム**（SVD等）を使用できるように実装してください。
    *   LLSがSLSのデータを直接受け取るわけではありません。「同じロジック」を両者で使えるようにすることが目的です。
    *   チャネル行列 $H$ を入力とし、プリコーディング行列 $W$ を返す関数/クラスとして実装してください。

### 2. `lls_scripts/run_papr_sim.py` および `components/pusch_model.py` の実装

#### 2.1 `PUSCHCommunicationModel` クラスの実装 (`components/pusch_model.py`) (推奨パス)
`5G_NR_PUSCH.ipynb` の内容をクラス化します。

*   **入力**: シミュレーション設定（Configクラス等）
*   **機能**:
    *   `PUSCHConfig` の初期化と設定。
    *   `PUSCHTransmitter`, `PUSCHReceiver` のインスタンス化。
    *   チャネルモデルの適用 (PAPR計測では不要かもしれないが、BLER用には必要)。
    *   **信号生成メソッド**: 時間領域信号 $x$ を生成して返す。
    *   **PAPR計測メソッド**: $x$ からPAPRを計算する。

#### 2.2 DFTS-OFDM (Transform Precoding) の手動実装
Sionna の `PUSCHConfig.transform_precoding` は、有効にすると MCS テーブルの参照先が変わってしまう等の問題があるため、以下の対応を行ってください。

*   **方針**:
    *   `PUSCHConfig.transform_precoding` は **OFF (0, CP-OFDM)** のままにするか、または MCS テーブルが正しく参照されるよう注意深く制御する。
    *   **DFT拡散 (DFT Spreading) を手動で実装する** ことを推奨します。
        *   `PUSCHTransmitter` の出力直前、または `PUSCHTransmitter` を継承/ラップして、変調シンボルがリソースグリッドにマッピングされる前に DFT を適用する処理を追加する。
        *   もしくは、Sionna が提供するコンポーネントを組み合わせて DFT-s-OFDM の信号生成フローを再現する。
    *   **電力正規化 (Power Normalization)**: DFT 適用前後で信号電力が変わらないよう、正規化係数 ($1/\sqrt{N_{DFT}}$ 等) を適切に適用すること。

### 3. PAPR シミュレーション実行 (`lls_scripts/run_papr_sim.py`)

*   `PUSCHCommunicationModel` を使用して以下を実行。
*   **パラメータスイープ**:
    *   Waveform: CP-OFDM vs DFT-s-OFDM (自作ロジック)
    *   Modulation: QPSK, 16QAM, 64QAM, 256QAM
    *   Rank: 1 (DFT-s-OFDMは通常Rank 1), または CP-OFDMなら 2~4
*   **オーバーサンプリング**:
    *   生成された信号を 4倍以上 にオーバーサンプリングして PAPR を計測。
*   **出力**: `mpr_table.csv`

## 成果物
1.  `components/precoder.py`: 共有プリコーダロジック
2.  `components/pusch_model.py`: `PUSCHCommunicationModel` クラス
3.  `lls_scripts/run_papr_sim.py`: 実行スクリプト
4.  `mpr_table.csv`: MPRテーブル
