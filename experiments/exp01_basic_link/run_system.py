import sys
import os

# --- プロジェクトルートをパスに追加 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# sys.path.append(project_root)
# ----------------------------------------

import wsim.rt.configs as cfg
from wsim.rt.runner import SionnaRunner

def get_scenarios():
    """実験シナリオ定義"""
    return [
        cfg.MasterConfig(
            exp_category="exp01_basic_link", # 実験テーマフォルダ名と合わせるのが通例だが、あえて同じカテゴリ内のsystem simとして扱う
            run_name="system_test_01",
            antenna=cfg.AntennaConfig(pattern="3gpp-3d", tilt_angle=5.0)
        )
    ]

if __name__ == "__main__":
    results_dir = os.path.join(current_dir, "results")

    for conf in get_scenarios():
        runner = SionnaRunner(conf)
        # System Level Simulation実行
        runner.run_system_level(results_dir)
