import os
import sys
import json
import dataclasses
import numpy as np
import tensorflow as tf
# Sionna imports
import sionna
try:
    from sionna.phy.channel.tr38901 import UMa, UMi, Antenna, AntennaArray, PanelArray
    from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
except ImportError:
    # Fallback or alternative locations if 1.2.1 struct is different
    from sionna.phy.channel.tr38901 import UMa, UMi
    from sionna.rt import Antenna, AntennaArray, PanelArray
    from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mapping import Mapper, Demapper
# utils might be in sionna.utils (if exposed by package) or sionna.phy.utils
try:
    # Attempt import from sionna.utils (standard)
    from sionna.utils import ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
except ImportError:
    # Fallback to sionna.phy.utils
    from sionna.phy.utils import ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank


# プロジェクトルートの解決
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.sls_configs import SLSMasterConfig

class MySLSRunner:
    """
    SLS (System Level Simulation) 用のランナー
    Sionna End-to-End Exampleをベースに実装
    """
    def __init__(self, config: SLSMasterConfig):
        self.c = config
        self.batch_size = 1 # SLSでは通常1

        # Simulation objects
        self.channel_model = None
        self.rg = None
        self.stream_manager = None
        self.ut_array = None
        self.bs_array = None
        self.topology = None # Placeholder if using explicit topology class

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

    def setup_system(self):
        """OFDM Resource Grid & Stream Management"""
        print("  [Step] Configuring System (Resource Grid)...")
        # 1. Resource Grid
        # Using configuration values
        self.rg = ResourceGrid(num_ofdm_symbols=self.c.carrier.num_ofdm_symbols,
                               fft_size=self.c.carrier.fft_size,
                               subcarrier_spacing=self.c.carrier.subcarrier_spacing,
                               num_tx=1, # Updated dynamically later based on topology if needed or fixed per UT
                               num_streams_per_tx=1,
                               cyclic_prefix_length=0, # Simplified
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=[2, 11])

    def setup_topology(self):
        """
        Topology Setup: BS and UT placement
        Using standard hexagonal grid logic or Sionna components
        """
        print("  [Step] Setting up Topology...")

        # Antenna Arrays
        # Example 3GPP Panel Array for BS
        # Correctly instantiating with positional arguments per Sionna v1.2.1
        self.bs_array = PanelArray(num_rows_per_panel=4,
                                   num_cols_per_panel=4,
                                   polarization="dual",
                                   polarization_type="VH",
                                   antenna_pattern="38.901",
                                   carrier_frequency=self.c.carrier.carrier_frequency,
                                   panel_vertical_spacing=3.0,
                                   panel_horizontal_spacing=3.0)

        # Single Antenna for UT
        # Correctly instantiating Antenna
        self.ut_array = Antenna(polarization="single",
                                polarization_type="V",
                                antenna_pattern="omni",
                                carrier_frequency=self.c.carrier.carrier_frequency)

        # Note: In a full SLS, we would define positions here.
        # For this template, we will rely on the Channel Model's internal topology generation
        # or simplified random drops if using the basic 3GPP channel class directly.
        # But UMa/UMi expects input positions or scenario configuration.

        # Simplified: Define just the counts for now, positions generated during channel usage or loop
        self.num_bs = self.c.topology.num_sites * self.c.topology.num_sectors
        self.num_ut = self.num_bs * self.c.topology.num_ues_per_sector

        print(f"    Created arrays. Num BS: {self.num_bs}, Num UT: {self.num_ut}")

    def setup_channel(self):
        """3GPP Channel Model Configuration"""
        print("  [Step] Configuring Channel Model...")

        # UMa Channel
        # Assuming Uplink for simplicity as per some tutorials, or Downlink
        self.channel_model = UMa(carrier_frequency=self.c.carrier.carrier_frequency,
                                 o2i_model="low",
                                 ut_array=self.ut_array,
                                 bs_array=self.bs_array,
                                 direction="uplink",
                                 enable_pathloss=True,
                                 enable_shadow_fading=True)

    def run_simulation(self):
        """
        Main Simulation Loop
        """
        print("  [Step] Starting Simulation Loop...")

        # 1. Topology Generation / Drop
        # For UMa, we need to set topology manually usually?
        # Or use channel_model.set_topology if available (it varies by version).
        # Standard way: define locations manually.

        # Randomly drop UTs and BSs (Simplified Hexagonal approximation)
        # Using a simple random drop for template purposes
        # BS at (0,0) and others around?
        # Let's generate random positions for demonstration

        # BS Positions (e.g., just one site for basic check or multi-site code)
        isd = self.c.topology.inter_site_distance
        bs_locs = []
        # Simple 1-tier hex grid logic could go here. For now, 1 central BS.
        bs_locs.append([0, 0, self.c.topology.bs_height])
        # Add more if num_sites > 1...

        # UT Positions (Randomly around BS)
        ut_locs = []
        min_dist = self.c.topology.min_dist
        max_dist = isd / 2
        for _ in range(self.num_ut):
            # For 3GPP models, distances must be realistic (> 10m usually)
            # Default min_dist in config is usually 10/35m.
            r = np.sqrt(np.random.uniform(min_dist**2, max_dist**2))
            phi = np.random.uniform(0, 2*np.pi)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            ut_locs.append([x, y, self.c.topology.ue_height])

        # Convert to Tensor and Expand Dims for Batch
        bs_locs = tf.constant(bs_locs, dtype=tf.float32)
        bs_locs = tf.expand_dims(bs_locs, axis=0) # [1, num_bs, 3]

        ut_locs = tf.constant(ut_locs, dtype=tf.float32)
        ut_locs = tf.expand_dims(ut_locs, axis=0) # [1, num_ut, 3]

        # Set Topology
        # Check Sionna API: channel.set_topology(ut_locs, bs_locs, ut_orientations, bs_orientations, ut_velocities, in_state)

        # Orientations (zeros)
        ut_orn = tf.zeros((self.batch_size, self.num_ut, 3), dtype=tf.float32)
        bs_orn = tf.zeros((self.batch_size, self.num_bs, 3), dtype=tf.float32)

        # User State (Indoor/Outdoor) - Required for UMa
        # 0: Outdoor, 1: Indoor. Assuming all outdoor for simplicity.
        # Shape: [batch_size, num_ut]
        in_state = tf.zeros((self.batch_size, self.num_ut), dtype=tf.bool)

        # UT Velocities - Required
        # Shape: [batch_size, num_ut, 3]
        ut_vel = tf.zeros((self.batch_size, self.num_ut, 3), dtype=tf.float32)

        self.channel_model.set_topology(ut_loc=ut_locs,
                                        bs_loc=bs_locs,
                                        ut_orientations=ut_orn,
                                        bs_orientations=bs_orn,
                                        ut_velocities=ut_vel,
                                        in_state=in_state)

        # 2. Time Evolution Loop
        # Generate Channel Response (CIR)
        print("    Generating channel coefficients (CIR)...")
        # Use sampling frequency inferred from ResourceGrid or Config
        sampling_freq = self.c.carrier.subcarrier_spacing * self.c.carrier.fft_size # e.g. 30e3 * 1024 ~ 30MHz
        num_time_samples = self.c.carrier.num_ofdm_symbols # Number of time steps? For SystemLevel, it's duration.

        # Get CIR (a, tau)
        cir = self.channel_model(num_time_samples=num_time_samples,
                                 sampling_frequency=sampling_freq)

        # 3. Frequency Domain Conversion
        print("    Converting to Frequency Domain...")
        frequencies = subcarrier_frequencies(self.c.carrier.fft_size, self.c.carrier.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)

        print(f"    Channel Shape: {h_freq.shape}")

        return h_freq

    def collect_metrics(self):
        """Calculate and save metrics"""
        print("  [Step] Collecting Metrics...")
        # Placeholder
        print("    (Metrics collected)")

    def run_system_level(self, base_result_dir: str):
        print(f"--> [SLS] Running: {self.c.exp_category} / {self.c.run_name}")
        save_dir = self._prepare_save_dir(base_result_dir, "system")
        self._save_snapshot(save_dir)

        self.setup_system()
        self.setup_topology()
        self.setup_channel()
        channel_output = self.run_simulation()
        self.collect_metrics()

        print(f"    Saved results to {save_dir}")
