# Agent B (SLS担当) 作業指示書

あなたは **SLS (System Level Simulator)** の実装担当です。
`../sionna-sls` ワークスペースで作業を行ってください。

## 目的
LLSが生成した MPR テーブルを活用した電力制御 ($P_{cmax}$ 制約) と、現実的な MCS (Modulation and Coding Scheme) テーブルに基づいたリンクアダプテーションを実装し、より高精度なシステムスループット評価を行うことです。

## タスク詳細

### 1. `components/link_adaptation.py` の実装 (または改修)
既存の Shannon 容量ベース (`WaterFillingLinkAdaptation`) ではなく、5G NR 標準の MCS テーブルを使用するクラスを作成します。

*   **クラス名**: `MCSLinkAdaptation` (新規)
*   **参照**: `sionna.phy.nr.PUSCHConfig`, `TBConfig`
*   **処理**:
    1.  **Effective SINR**: MIMO受信後のSINRを入力とする。
    2.  **MCS Lookup**: SINR と BLERターゲット (例: 10%) から、最適な MCS Index を選択する。
        *   Sionna に `select_mcs` のようなユーティリティがあれば活用する。なければ、SINR-BLER曲線 (AWGNベースの近似) を用いてマッピングする。
    3.  **TBS & Throughput**:
        *   選択された MCS Index と割り当て RB 数から Transport Block Size (TBS) を計算。
        *   `Throughput = TBS / SlotDuration * (1 - BLER)`

### 2. `components/mpr_model.py` の実装
*   **入力**: `mpr_table.csv` (Agent Aが作成。無い場合はダミーを作成して進める)
*   **機能**: UEの設定 (Waveform, Rank) に応じた MPR 値 (dB) を返す。

### 3. `components/power_control.py` の実装
*   **ロジック**: $P_{tx} = \min(P_{cmax}, P_{open\_loop})$
    *   $P_{cmax} = P_{power\_class} - MPR$
    *   $P_{open\_loop} = P_0 + 10 \log_{10}(M_{RB}) + \alpha \cdot PL$
*   **入力**: Pathloss $PL$, 割り当て帯域 $M_{RB}$, パラメータ $P_0, \alpha$, 最大電力 $P_{power\_class}$, MPR。

### 4. `simulator.py` の統合
*   `simulator.py` 内の `simulate_slot` ループ内で上記コンポーネントを呼び出す。
    1.  チャネル推定 -> `precoder` (Agent Aの成果物がまだなら理想SVD)
    2.  `power_control` -> $P_{tx}$ 決定 -> 受信信号電力スケーリング
    3.  干渉計算 -> SINR
    4.  `link_adaptation` -> MCS決定 -> スループット算出

## 成果物
1.  `components/link_adaptation.py`: MCS対応版
2.  `components/mpr_model.py`: MPRローダー
3.  `components/power_control.py`: 電力制御ロジック
4.  `simulator.py`: 改修版
