# Agent A (LLS担当) 作業指示書

あなたは **LLS (Link Level Simulator)** の実装担当です。
`../sionna-lls` ワークスペースで作業を行ってください。

## 目的
Sionna のチュートリアル `5G_NR_PUSCH.ipynb` の内容をベースに、クラス継承を活用して拡張性を確保しつつ、PAPR (Peak-to-Average Power Ratio) を評価して MPR テーブルを作成することです。

## タスク詳細

### 1. `components/precoder.py` の実装
*   **役割**: プリコーディング行列生成ロジックの提供 (ライブラリ)。
*   **仕様**:
    *   SLS/LLS共通で使用するアルゴリズム（SVD等）を実装。
    *   データの受け渡し用ではなく、計算ロジックを共有するためのモジュール。

### 2. `components/pusch_transmitter_wrapper.py` の実装 (継承アプローチ)
Sionna の `PUSCHTransmitter` を直接修正するのではなく、**継承して機能を拡張**します。

*   **クラス**: `HybridPUSCHTransmitter(PUSCHTransmitter)`
*   **DFTS-OFDM (Transform Precoding) の実装**:
    *   `PUSCHConfig` の `transform_precoding` フラグは MCS テーブルの挙動を変えてしまうため、**Config 上は OFF (CP-OFDM扱い)** に設定し、このクラス内部で手動で DFT 拡散を適用します。
    *   **実装ポイント**:
        *   親クラスのメソッド（例: `_generate_grid` や `call` 内部の変調シンボル生成部分）をオーバーライド、または出力直前の信号に対して DFT 処理を挟み込む。
        *   **電力正規化**: DFT 適用前後で電力が変わらないよう正規化 ($1/\sqrt{N_{DFT}}$) を徹底する。

### 3. `components/channel_models.py` の拡張
既存の `experiments/sls_end2end_hybrid_beam/components/channel_models.py` に新しいクラスを追加します。

*   **クラス**: `GenerateHybridBeamformingTimeChannel`
*   **ベース**: `ChunkedGenerateTimeChannel` (継承) および `GenerateHybridBeamformingOFDMChannel` (ロジック参考)
*   **仕様**:
    *   時間領域チャネル ($h_{time}$) 生成機能を持つ。
    *   **Generation と Precoding の分離**:
        *   チャネル生成 ($a, \tau$) と、アナログビームフォーミング ($W_{RF}, A_{RF}$) の適用を明確に分離して実装、またはメソッドを分ける。
        *   `GenerateHybridBeamformingOFDMChannel` の `_apply_weights` に相当する処理を、時間領域チャネル (CIR) または Time Channel に適用できるようにポーティングする。

### 4. `components/pusch_model.py` (PUSCHCommunicationModel)
チュートリアルのフローをカプセル化します。

*   **クラス**: `PUSCHCommunicationModel`
*   **メンバ**:
    *   `HybridPUSCHTransmitter` (上記で作成したクラスを使用)
    *   `GenerateHybridBeamformingTimeChannel` (チャネルモデル)
    *   `PUSCHReceiver`
*   **メソッド**:
    *   `generate_waveform()`: 送信波形生成。
    *   `compute_papr()`: PAPR計測。

### 5. シミュレーション実行 (`lls_scripts/run_papr_sim.py`)
*   `PUSCHCommunicationModel` を使用してパラメータスイープを実行。
*   Waveform (CP vs DFTS), Modulation, Rank を変更しながら `mpr_table.csv` を生成。

## 成果物
1.  `components/precoder.py`
2.  `components/pusch_transmitter_wrapper.py`
3.  `components/channel_models.py` (追記)
4.  `components/pusch_model.py`
5.  `lls_scripts/run_papr_sim.py`
6.  `mpr_table.csv`
