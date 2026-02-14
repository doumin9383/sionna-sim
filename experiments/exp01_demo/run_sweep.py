import sys
import os
import copy
# Add project root to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from wsim.rt.configs import (
    SimulationConfig, SceneConfig, PlanarArrayConfig, TransmitterConfig,
    ReceiverConfig, RayTracingConfig
)
from wsim.rt.runner import SionnaRunner

def main():
    # Base Configuration Components
    scene_cfg = SceneConfig(frequency=3.5e9, synthetic_array=True)
    rt_cfg = RayTracingConfig(max_depth=5, samples_per_src=10000) # Fast run
    rx_array = PlanarArrayConfig(num_rows=1, num_cols=1)
    rx1 = ReceiverConfig(name="rx1", position=[50, 0, 1.5], antenna_array=rx_array)

    # Sweep Parameters: Varrying TX Array Size
    configs = []

    # Case 1: 4 Antennas
    tx_array_4 = PlanarArrayConfig(num_rows=1, num_cols=4)
    tx_4 = TransmitterConfig(name="tx_4", position=[0, 0, 10], antenna_array=tx_array_4)

    cfg_1 = SimulationConfig(
        scene=scene_cfg,
        transmitters=[tx_4],
        receivers=[rx1],
        ray_tracing=rt_cfg,
        output_dir="results/sweep_case_4_antennas"
    )
    configs.append(cfg_1)

    # Case 2: 8 Antennas
    tx_array_8 = PlanarArrayConfig(num_rows=1, num_cols=8)
    tx_8 = TransmitterConfig(name="tx_8", position=[0, 0, 10], antenna_array=tx_array_8)

    cfg_2 = SimulationConfig(
        scene=scene_cfg,
        transmitters=[tx_8],
        receivers=[rx1],
        ray_tracing=rt_cfg,
        output_dir="results/sweep_case_8_antennas"
    )
    configs.append(cfg_2)

    # Execution Loop
    print(f"Starting sweep with {len(configs)} configurations...")

    for i, cfg in enumerate(configs):
        print(f"\n--- Running Config {i+1}/{len(configs)}: {cfg.output_dir} ---")
        runner = SionnaRunner(cfg)
        runner.run()

    print("\nSweep completed.")

if __name__ == "__main__":
    main()
