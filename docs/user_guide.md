# Sionnaシミュレーション ユーザーガイド

このリポジトリは、Sionna (Ray Tracing) を用いた6G物理層シミュレーションのための環境です。

## 1. はじめに

本環境は、設定（Config）、実行（Runner）、実験（Experiments）を明確に分離し、パラメータスイープや再現実験を容易に行えるように設計されています。

## 2. クイックスタート

### 2.1 動作確認（デモ）

単一の構成でシミュレーションを実行します。

```bash
python experiments/exp01_demo/run_single.py
```

実行後、`results/` ディレクトリに結果ファイルと設定スナップショットが生成されます。

### 2.2 パラメータスイープ

複数の設定を一括で実行します。

```bash
python experiments/exp01_demo/run_sweep.py
```

## 3. 設定のカスタマイズ

シミュレーションの設定は `libs/my_configs.py` に定義されたデータクラスを通じて行います。IDEの補完機能を活用してパラメータを設定してください。

### 主要なConfigクラス
- **SimulationConfig**: 全体設定のルート
- **Scene**: シーン（3Dモデル）の設定
- **StreamManagement**: 送受信アンテナ間のストリーム割り当て
- **ResourceGrid**: 時間・周波数リソースの設定

設定を変更して新しい実験を行う場合は、`experiments/` 以下に新しいフォルダ（例: `exp02_my_test`）を作成し、スクリプトをコピーして編集することを推奨します。

## 4. トラブルシューティング

シミュレーションがうまく動かない場合は、`results/` フォルダに出力される `config_snapshot.py` を確認してください。これは実行時の設定をそのままPythonコードとして保存したものです。
