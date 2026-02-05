import os
import sys
import json
import dataclasses
from abc import ABC, abstractmethod
from typing import Any

# libs がパス解決されている前提でのインポート
# 実験スクリプト側で sys.path.append される
try:
    from libs.my_configs import MasterConfig
except ImportError:
    # 開発中の単体テスト用などのフォールバック
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from libs.my_configs import MasterConfig

class SionnaRunner:
    """
    Sionnaシミュレーション実行の基底ランナー
    設定の保持、結果保存ディレクトリの解決、スナップショット保存を担当する
    """
    def __init__(self, config: MasterConfig):
        self.c = config

    def _prepare_save_dir(self, base_result_dir: str, mode: str) -> str:
        """
        保存先ディレクトリを作成する
        path: {base_result_dir}/{exp_category}/{results}/{run_name}_{mode}
        """
        # ユーザー要望に基づき experiments/expXX/results/runXX 形式もサポートできるよう
        # 引数 base_result_dir は呼び出し元で調整可能とするが、
        # ここでは単純に join する実装とする。

        # 例: experiments/exp01/results/tilt_0_link
        save_dir = os.path.join(base_result_dir, f"{self.c.run_name}_{mode}")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _save_snapshot(self, save_dir: str):
        """
        現在の設定を再現可能なPythonコードとして保存する
        """
        snapshot_path = os.path.join(save_dir, "config_snapshot.py")

        with open(snapshot_path, "w") as f:
            f.write("import sys\nimport os\n")
            f.write("# プロジェクトルートへのパス解決 (環境に合わせて調整が必要になる場合あり)\n")
            f.write("sys.path.append(os.path.abspath('../../../..'))\n\n")

            f.write("from libs.my_configs import MasterConfig, AntennaConfig, WaveformConfig, SimulationParameters\n\n")

            f.write("# Reconstructed Configuration\n")
            f.write(f"config = {repr(self.c)}\n")

        # JSON形式でも保存（可読性のため）
        json_path = os.path.join(save_dir, "config.json")
        try:
            with open(json_path, "w") as f:
                json.dump(dataclasses.asdict(self.c), f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save JSON config. {e}")

    def run_link_level(self, base_result_dir: str):
        """
        リンクレベルシミュレーションを実行する (具象メソッド)
        """
        print(f"--> [Link-Level] Running: {self.c.exp_category} / {self.c.run_name}")
        save_dir = self._prepare_save_dir(base_result_dir, "link")
        self._save_snapshot(save_dir)

        # ここにSionnaのリンクレベルシミュレーション構築ロジックが入る
        # self._execute_link_sim(save_dir) (未実装)
        print(f"    Saved snapshot to {save_dir}")

    def run_system_level(self, base_result_dir: str):
        """
        システムレベルシミュレーションを実行する (具象メソッド)
        """
        print(f"--> [System-Level] Running: {self.c.exp_category} / {self.c.run_name}")
        save_dir = self._prepare_save_dir(base_result_dir, "system")
        self._save_snapshot(save_dir)

        # ここにSionnaのシステムレベルシミュレーション構築ロジックが入る
        # self._execute_system_sim(save_dir) (未実装)
        print(f"    Saved snapshot to {save_dir}")
