import os
import sys
import tensorflow as tf
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from experiments.hybrid_beamforming.sls.my_configs import HybridSLSConfig
from experiments.hybrid_beamforming.sls.simulator import HybridSystemSimulator
from sionna.phy.channel.tr38901 import PanelArray


from sionna.phy.ofdm import ResourceGrid
from libs.my_configs import ResourceGridConfig


class TestRBGGranularity(unittest.TestCase):
    def setUp(self):
        # Configure minimal simulation parameters
        self.bs_array = PanelArray(
            num_rows_per_panel=4,
            num_cols_per_panel=4,
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=3.5e9,
        )
        self.ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=3.5e9,
        )

        # Create ResourceGrid
        rg_config = ResourceGridConfig(
            num_ofdm_symbols=14, fft_size=64, subcarrier_spacing=30e3
        )
        self.rg = ResourceGrid(
            num_ofdm_symbols=rg_config.num_ofdm_symbols,
            fft_size=rg_config.fft_size,
            subcarrier_spacing=rg_config.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=6,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
        )

    def test_granularity_enabled(self):
        print("\n--- Testing RBG Granularity Enabled ---")
        rbg_size_rb = 4
        rbg_size_sc = rbg_size_rb * 12

        config = HybridSLSConfig(
            batch_size=1,
            num_ut_per_sector=1,
            num_rings=1,
            bs_array=self.bs_array,
            ut_array=self.ut_array,
            resource_grid=self.rg,
            precoding_granularity="Narrowband",  # Doesn't matter for this test directly
            rbg_size_rb=rbg_size_rb,
            use_rbg_granularity=True,  # ENABLE MODE
            output_dir="./test_results",
        )

        simulator = HybridSystemSimulator(config)

        # 1. Verify Configuration Propagation
        self.assertTrue(simulator.channel_interface.use_rbg_granularity)
        self.assertTrue(simulator.channel_interface.hybrid_channel.use_rbg_granularity)
        self.assertEqual(
            simulator.channel_interface.hybrid_channel.rbg_size, rbg_size_sc
        )

        # 2. Verify Channel Output Shape (Reduced Dimension)
        # get_full_channel_info calls hybrid_channel(batch_size)
        # Check num_sc dimension
        h, s, _, _ = simulator.channel_interface.get_full_channel_info(1)
        # h shape: [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]
        # index 4 is num_sc (which should be num_rbgs now)

        print("Singular Values (s):", s.numpy())
        print("Max Singular Value:", tf.reduce_max(s).numpy())
        print("Min Singular Value:", tf.reduce_min(s).numpy())

        total_sc = config.resource_grid.num_effective_subcarriers
        expected_rbgs = total_sc // rbg_size_sc

        print(
            f"Total SC: {total_sc}, RBG Size: {rbg_size_sc}, Expected RBGs: {expected_rbgs}"
        )
        print(f"Channel Shape: {h.shape}")

        self.assertEqual(
            h.shape[4],
            expected_rbgs,
            "Channel frequency dimension should match number of RBGs",
        )

        # 3. Verify Simulation Run (Throughput calculation)
        try:
            results = simulator(num_slots=1, tx_power_dbm=23.0)
            print("Simulation run successful.")
            mean_val = tf.reduce_mean(results).numpy()
            print(f"Mean Throughput: {mean_val}")
        except Exception as e:
            print(f"Simulation failed with error: {e}")

    def test_default_mode(self):
        print("\n--- Testing Default Mode (Full Granularity) ---")
        config = HybridSLSConfig(
            batch_size=1,
            num_ut_per_sector=1,
            num_rings=1,
            bs_array=self.bs_array,
            ut_array=self.ut_array,
            resource_grid=self.rg,
            use_rbg_granularity=False,  # DISABLE MODE
            output_dir="./test_results",
        )

        simulator = HybridSystemSimulator(config)

        # Verify Dimension
        h, _, _, _ = simulator.channel_interface.get_full_channel_info(1)
        total_sc = config.resource_grid.num_effective_subcarriers

        print(f"Total SC: {total_sc}")
        print(f"Channel Shape: {h.shape}")

        self.assertEqual(
            h.shape[4],
            total_sc,
            "Channel frequency dimension should match total subcarriers",
        )

        try:
            results = simulator(num_slots=1, tx_power_dbm=23.0)
            print("Simulation run successful.")
            mean_val = tf.reduce_mean(results).numpy()
            print(f"Mean Throughput: {mean_val}")
        except Exception as e:
            print(f"Simulation failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
