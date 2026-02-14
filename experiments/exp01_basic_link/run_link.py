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
    scenarios = []

    # Case 1: Default
    scenarios.append(cfg.MasterConfig(
        exp_category="exp01_basic_link",
        run_name="case_default",
        waveform=cfg.WaveformConfig(modulation="QAM16")
    ))

    # Case 2: 64QAM
    scenarios.append(cfg.MasterConfig(
        exp_category="exp01_basic_link",
        run_name="case_64qam",
        waveform=cfg.WaveformConfig(modulation="QAM64")
    ))

    return scenarios

if __name__ == "__main__":
    # 出力先: experiments/exp01_basic_link/results
    results_dir = os.path.join(current_dir, "results")

    print(f"Project Root: {project_root}")
    print(f"Results Dir:  {results_dir}")

    for conf in get_scenarios():
        runner = SionnaRunner(conf)
        # Link Level Simulation実行
        runner.run_link_level(results_dir)
