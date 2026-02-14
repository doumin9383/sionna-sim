import sys
import os

# --- プロジェクトルートをパスに追加 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# sys.path.append(project_root)
# ----------------------------------------

from wsim.sls.configs import (
    SLSMasterConfig,
    CarrierConfig,
    TopologyConfig,
    ChannelConfig,
    AntennaConfig,
)
from experiments.sls_end2end.runner import MySLSRunner


def get_scenarios():
    """実験シナリオ定義"""
    return [
        SLSMasterConfig(
            exp_category="sls_end2end",
            run_name="test_run_01",
            # SLS固有設定
            carrier=CarrierConfig(
                subcarrier_spacing=30e3,
                bandwidth=2.16e6,  # 72 subcarriers * 30kHz
                fft_size=72,  # Small for testing
                carrier_frequency=3.5e9,
            ),
            topology=TopologyConfig(
                inter_site_distance=500.0,
                num_sites=1,  # Simplified for template verification
                num_sectors=1,  # Simplified
                num_ues_per_sector=2,  # Reduced for speed
            ),
            channel=ChannelConfig(model_name="UMa", los_probability=None),
            # 共通設定
            antenna=AntennaConfig(
                pattern="3gpp", num_rows=4, num_cols=4, polarization="VH"
            ),
        )
    ]


if __name__ == "__main__":
    results_dir = os.path.join(current_dir, "results")

    for conf in get_scenarios():
        runner = MySLSRunner(conf)
        runner.run_system_level(results_dir)
