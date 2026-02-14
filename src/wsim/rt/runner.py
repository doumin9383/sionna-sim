import os
import time
import dataclasses
from dataclasses import asdict
import tensorflow as tf
import sionna
from sionna.rt import Scene, PlanarArray, Transmitter, Receiver, Camera, PathSolver
from wsim.rt.configs import SimulationConfig
import numpy as np

class SionnaRunner:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.scene = None
        self._setup_output_dir()

    def _setup_output_dir(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        # Save config snapshot
        snapshot_path = os.path.join(self.cfg.output_dir, "config_snapshot.py")
        with open(snapshot_path, "w") as f:
            f.write(f"from wsim.rt.configs import *\nimport numpy as np\n\n")
            f.write(f"config = {repr(self.cfg)}")

    def load_scene(self):
        """Loads and configures the Sionna Scene"""
        # Create Scene
        # Note: Scene() loading logic is complex, often involves load_scene(filename)
        # Here we assume loading from file if provided, or empty
        if self.cfg.scene.filename:
            self.scene = sionna.rt.load_scene(self.cfg.scene.filename)
        else:
            self.scene = Scene() # Default empty scene or use other params

        # Configure scene parameters
        self.scene.frequency = self.cfg.scene.frequency
        self.scene.synthetic_array = self.cfg.scene.synthetic_array

        # Add Transmitters
        for tx_cfg in self.cfg.transmitters:
            # Create Antenna Array
            # unpack PlanarArrayConfig to PlanarArray
            array_params = asdict(tx_cfg.antenna_array)
            # 'pattern' is a string in config, PlanarArray expects string or callable.
            # PlanarArray(..., pattern='iso', ...) works.

            antenna_array = PlanarArray(**array_params)

            # Set scene's tx_array before adding the transmitter
            # Transmitter uses the currently set scene.tx_array designated by the scene
            # or we need to ensure it's linked.
            # In Sionna RT, standard flow is scene.tx_array = ..., scene.add(Transmitter(...))
            self.scene.tx_array = antenna_array

            # Create Transmitter
            # Setup params, excluding 'antenna_array' from dict as we pass the object
            tx_params = asdict(tx_cfg)
            del tx_params['antenna_array'] # remove nested config dict

            tx = Transmitter(**tx_params)
            self.scene.add(tx)

        # Add Receivers
        for rx_cfg in self.cfg.receivers:
            array_params = asdict(rx_cfg.antenna_array)
            antenna_array = PlanarArray(**array_params)

            # Set scene's rx_array
            self.scene.rx_array = antenna_array

            rx_params = asdict(rx_cfg)
            del rx_params['antenna_array']

            rx = Receiver(**rx_params)
            self.scene.add(rx)

    def run_ray_tracing(self):
        """Executes compute_paths using PathSolver"""
        if self.scene is None:
            raise ValueError("Scene not loaded. Call load_scene() first.")

        rt_params = asdict(self.cfg.ray_tracing)

        solver = PathSolver()
        paths = solver(self.scene, **rt_params)

        # Example processing: coverage map or path collection
        # For now, just return paths
        return paths

    def run(self):
        print(f"Starting simulation...")
        self.load_scene()
        paths = self.run_ray_tracing()
        print(f"Ray tracing computed. Paths object: {type(paths)}")
        try:
             # Attempt to access 'a' or 'cir' to confirm
             print(f"Paths coefficients 'a' type: {type(paths.a)}")
             print(f"Paths shape (if tensor): {paths.a.shape if hasattr(paths.a, 'shape') else 'N/A'}")
        except Exception as e:
             print(f"Could not inspect paths details: {e}")

        # Save results/metrics (dummy implementation)
        # paths.save(...)
        return paths
