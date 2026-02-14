import os
import sys
import json
import pickle
import dataclasses
import numpy as np
import tensorflow as tf

# Sionna imports
import sionna
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel.tr38901 import Antenna, PanelArray

# Local components import
# Assuming components package is reachable.
# Since runner.py is in experiments/sls_end2end/, we can import from .components
try:
    from .components.sls_simulaiton import SystemLevelSimulator
except ImportError:
    # If running as script, path might need adjustment
    # sys.path.append(os.path.dirname(__file__))
    from components.sls_simulaiton import SystemLevelSimulator

# Config import
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from wsim.sls.configs import SLSMasterConfig


class MySLSRunner:
    """
    SLS Runner wrapping the Tutorial-based SystemLevelSimulator
    """

    def __init__(self, config: SLSMasterConfig):
        self.c = config
        self.batch_size = 1

        # Instantiate objects needed for Simulator
        self._setup_resource_grid()
        self._setup_arrays()

    def _setup_resource_grid(self):
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.c.carrier.num_ofdm_symbols,
            fft_size=self.c.carrier.fft_size,
            subcarrier_spacing=self.c.carrier.subcarrier_spacing,
            num_tx=1,  # Placeholder, updated in Sim
            num_streams_per_tx=1,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
        )

    def _setup_arrays(self):
        # BS: PanelArray (using correct 1.2.1 args)
        self.bs_array = PanelArray(
            num_rows_per_panel=4,
            num_cols_per_panel=4,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=self.c.carrier.carrier_frequency,
            panel_vertical_spacing=3.0,
            panel_horizontal_spacing=3.0,
        )

        # UT: Antenna
        self.ut_array = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.c.carrier.carrier_frequency,
        )

    def _prepare_save_dir(self, base_result_dir: str, mode: str) -> str:
        """保存先ディレクトリを作成する"""
        save_dir = os.path.join(base_result_dir, f"{self.c.run_name}_{mode}")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _save_snapshot(self, save_dir: str):
        """現在の設定を保存する"""
        json_path = os.path.join(save_dir, "config.json")
        try:
            with open(json_path, "w") as f:
                json.dump(dataclasses.asdict(self.c), f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save JSON config. {e}")

    def run_system_level(self, base_result_dir: str):
        print(
            f"--> [SLS] Running: {self.c.exp_category} / {self.c.run_name} (Tutorial Logic)"
        )
        save_dir = self._prepare_save_dir(base_result_dir, "system")
        self._save_snapshot(save_dir)

        # Instantiate Simulator
        # Map config to args
        # num_rings: 1 -> 7 sites (approx)
        num_rings = 1

        sim = SystemLevelSimulator(
            batch_size=self.batch_size,
            num_rings=num_rings,
            num_ut_per_sector=self.c.topology.num_ues_per_sector,
            carrier_frequency=self.c.carrier.carrier_frequency,
            resource_grid=self.rg,
            scenario=self.c.channel.model_name.lower(),  # "uma", "umi", "rma"
            direction="uplink",  # Default to uplink
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            bs_max_power_dbm=43.0,
            ut_max_power_dbm=23.0,
            coherence_time=20,  # Slots
        )

        # Run Simulation
        print("  [Step] Starting Simulation Loop...")
        # num_slots to simulate
        num_slots = 20  # Example duration

        # OLLA params
        alpha_ul = 0.6
        p0_dbm_ul = -80.0
        bler_target = 0.1
        olla_delta_up = 0.01

        hist = sim(
            num_slots=num_slots,
            alpha_ul=alpha_ul,
            p0_dbm_ul=p0_dbm_ul,
            bler_target=bler_target,
            olla_delta_up=olla_delta_up,
        )

        print("  [Step] Simulation Completed.")

        # Save Results (Pickle hist as it contains numpy arrays)
        # hist is a dictionary of arrays.
        results_path = os.path.join(save_dir, "history.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(hist, f)

        print(f"    Saved history to {results_path}")
