import sys
import os
import dataclasses
# Add project root to sys.path to enable 'from libs...' imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from wsim.rt.configs import (
    SimulationConfig, SceneConfig, PlanarArrayConfig, TransmitterConfig,
    ReceiverConfig, RayTracingConfig
)
from wsim.rt.runner import SionnaRunner

def main():
    # Define Configuration

    # 1. Scene
    # Note: Using default simple scene settings since we don't have a file yet
    scene_cfg = SceneConfig(
        frequency=3.5e9,
        synthetic_array=True
    )

    # 2. Transmitters
    tx_array = PlanarArrayConfig(
        num_rows=1, num_cols=4,
        pattern="iso", polarization="VH"
    )
    tx1 = TransmitterConfig(
        name="tx1",
        position=[0, 0, 10],
        antenna_array=tx_array
    )

    # 3. Receivers
    rx_array = PlanarArrayConfig(num_rows=1, num_cols=1)
    rx1 = ReceiverConfig(
        name="rx1",
        position=[50, 0, 1.5],
        antenna_array=rx_array
    )

    # 4. Ray Tracing Params
    rt_cfg = RayTracingConfig(
        max_depth=5,
        samples_per_src=100000
        # Reduced samples for quick demo
    )

    # 5. Master Config
    sim_config = SimulationConfig(
        scene=scene_cfg,
        transmitters=[tx1],
        receivers=[rx1],
        ray_tracing=rt_cfg,
        output_dir="results/demo_single"
    )

    # Run Simulation
    runner = SionnaRunner(sim_config)
    paths = runner.run()

    print("Single run completed successfully.")

if __name__ == "__main__":
    main()
