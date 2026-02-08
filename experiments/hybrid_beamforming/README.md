# Hybrid Beamforming & Link Adaptation Simulation Guide

このディレクトリ (`experiments/hybrid_beamforming`) には、MIMOプリコーディング粒度と波形（CP-OFDM vs DFT-s-OFDM）がシステム性能に与える影響を評価するためのシミュレーション環境が含まれています。

シミュレーションは **Phase 1: LLS (Link Level Simulation)** と **Phase 2: SLS (System Level Simulation)** の2段階で構成されます。

## 前提条件

プロジェクトルートで仮想環境が有効であることを確認してください。
```bash
source .venv/bin/activate
# または
source .venv/bin/activate.fish
```
実行はプロジェクトルートから `uv run python3 ...` コマンドで行います。

---

## Phase 1: LLS (PAPR & MPR Evaluation)

物理層の特性（PAPR）を評価し、MPR（Maximum Power Reduction）テーブルを作成します。このテーブルはPhase 2で使用されます。

### 1. シミュレーション実行 (PAPR計算)
各波形・変調方式・ランク・粒度の組み合わせについてPAPRを計算し、MPRテーブルを生成します。

```bash
uv run python3 experiments/hybrid_beamforming/lls/run_papr_sim.py
```

*   **出力**:
    *   `lls/results/mpr_table.csv`: 計算されたMPRテーブル（SLSで使用）。
    *   `lls/results/ccdfs/`: 各パターンのPAPR CCDFグラフ。
    *   `lls/results/waveforms/`: 生成された時間波形の例。
    *   `lls/results/papr_ccdf_summary.png`: 全パターンの比較グラフ。

### 2. スループット解析 (理論値確認)
作成されたMPRテーブルに基づき、SNRに対する理論スループット（Spectral Efficiency）をプロットします。

```bash
uv run python3 experiments/hybrid_beamforming/lls/eval_throughput.py
```

*   **出力**:
    *   `lls/results/throughput_vs_snr.png`: 波形・粒度ごとのスループット比較。
    *   `lls/results/optimal_config_map.png`: SNRごとの最適な設定（波形・粒度）の遷移マップ。

---

## Phase 2: SLS (System Level Evaluation)

マルチセル環境において、Phase 1のMPRテーブルを利用しながらシステムスループットを評価します。

### 1. シミュレーション実行
ユーザをランダムに配置し、リンクアダプテーション（MCS/Rank選択）と電力制御を行いながらスループットを計測します。

```bash
uv run python3 experiments/hybrid_beamforming/sls/run_sim.py
```

*   **設定**: `sls/my_configs.py` でシナリオ、ユーザー数、リング数などを変更可能です。
*   **出力**:
    *   `sls/results/history.pkl`: シミュレーションの詳細履歴（Python Pickle形式）。
    *   `sls/results/simulation_results.csv`: スロットごとの平均スループット推移（簡易バックアップ）。

### 2. 結果解析・可視化
保存された履歴データ (`history.pkl`) を読み込み、詳細なグラフを生成します。

```bash
uv run python3 experiments/hybrid_beamforming/sls/analyze_results.py
```

*   **出力**:
    *   `sls/results/throughput_cdf.png`: ユーザスループットの累積分布関数 (CDF)。
    *   `sls/results/network_throughput_time_series.png`: システム全体のスループット時系列。
    *   `sls/results/mcs_distribution.png`: 選択されたMCSインデックスのヒストグラム。
    *   `sls/results/sinr_cdf.png`: 実効SINRのCDF。

---

## ディレクトリ構造

```text
experiments/hybrid_beamforming/
├── lls/                  # Phase 1: Link Level Simulation
│   ├── run_papr_sim.py   # PAPRシミュレーション実行スクリプト
│   ├── eval_throughput.py# MPRに基づく理論スループット解析
│   ├── components/       # LLS用コンポーネント (PUSCHモデル等)
│   └── results/          # LLS結果出力先
├── sls/                  # Phase 2: System Level Simulation
│   ├── run_sim.py        # SLS実行スクリプト
│   ├── analyze_results.py# SLS結果解析・可視化スクリプト
│   ├── simulator.py      # SLSメインループ実装
│   ├── components/       # SLS用コンポーネント (チャネル、スケジューラ等)
│   └── results/          # SLS結果出力先
├── shared/               # 共通コンポーネント
│   ├── channel_models/   # 共通チャネルモデル設定
│   └── ...
└── sim_spec.md           # シミュレーション仕様書
```
