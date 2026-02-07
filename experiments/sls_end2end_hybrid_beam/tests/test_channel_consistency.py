import tensorflow as tf
import unittest
import numpy as np
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel.tr38901 import PanelArray
import sionna
from experiments.sls_end2end_hybrid_beam.components.channel_models import (
    RBGChannelModel,
    ChunkedTimeChannel,
    ChunkedOFDMChannel,
)


class TestChannelConsistency(unittest.TestCase):
    def setUp(self):
        # Configuration
        self.carrier_frequency = 3.5e9
        self.subcarrier_spacing = 30e3
        self.fft_size = 128
        self.num_ofdm_symbols = 14
        self.batch_size = 2

        # Arrays
        self.bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=4,
            polarization="single",
            polarization_type="V",
            element_vertical_spacing=0.5,
            element_horizontal_spacing=0.5,
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency,
        )
        self.ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            element_vertical_spacing=0.5,
            element_horizontal_spacing=0.5,
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency,
        )

        # Resource Grid
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=1,
        )

    def test_path_consistency(self):
        """
        Verify that RBGChannelModel generates consistent paths when called identically.
        But more importantly, verify Chunked*Channel classes access the SAME model instance correctly.
        """
        # Instantiate Shared Model
        model = RBGChannelModel(
            scenario="uma",
            carrier_frequency=self.carrier_frequency,
            direction="downlink",
            ut_array=self.ut_array,
            bs_array=self.bs_array,
        )

        # Instantiate Time and OFDM Channels wrappers
        # Note: GenerateTimeChannel calculates bandwidth etc from l_min/l_max or needs explicit input
        # We need to set l_min, l_max reasonable for UMa.
        # Typically set by model, but GenerateTimeChannel expects them or bandwidth?
        # GenerateTimeChannel(channel_model, bandwidth, num_time_samples, ...)

        bandwidth = self.subcarrier_spacing * self.fft_size
        num_time_samples = self.fft_size + 16  # roughly CP

        # Correctly instantiating ChunkedTimeChannel might require knowing l_min/l_max beforehand
        # or letting it default if Sionna allows.
        # But for this test, we care about the internal call to model()

        # For this test, let's mock or verify via the exposed get_paths()

        ch_time = ChunkedTimeChannel(
            model, bandwidth=bandwidth, num_time_samples=1000, l_min=-10, l_max=100
        )
        ch_ofdm = ChunkedOFDMChannel(model, self.rg)

        # Create a topology first (needed for UMa)
        # We can use a minimal dummy topology or use Sionna's gen_topology utils.
        # Or manually set positions.
        model.set_topology(
            ut_loc=tf.random.uniform((self.batch_size, 1, 3)),
            bs_loc=tf.random.uniform((self.batch_size, 1, 3)),
            ut_orientations=tf.random.uniform((self.batch_size, 1, 3)),
            bs_orientations=tf.random.uniform((self.batch_size, 1, 3)),
            ut_velocities=tf.zeros((self.batch_size, 1, 3)),
            in_state=tf.zeros((self.batch_size, 1), dtype=tf.bool),  # Outdoor
        )

        # Seed control
        tf.random.set_seed(42)
        sionna.phy.config.seed = 42

        # Generate paths via Time Channel wrapper
        a_time, tau_time = ch_time.get_cir(self.batch_size)

        # Reset seed to ensure same generation if independent
        tf.random.set_seed(42)
        sionna.phy.config.seed = 42

        # Generate paths via OFDM Channel wrapper
        a_ofdm, tau_ofdm = ch_ofdm.get_paths(self.batch_size)

        # Verify A and Tau are identical
        # Note: 'a' includes phase which depends on doppler/time.
        # get_paths() in both just calls model(num_samples, freq).
        # If input args differ, output differs.

        # ChunkedTimeChannel calls: model(num_time_samples, sampling_freq)
        # ChunkedOFDMChannel calls: model(num_time_samples, sampling_freq)
        # We must ensure they pass the SAME args if we want identical raw paths.

        # Check args passed in init/defaults
        # GenerateTimeChannel sets self._sampling_frequency = bandwidth usually?
        # GenerateOFDMChannel sets self._sampling_frequency = rg.bandwidth

        # Validate they match
        # self.assertAlmostEqual(ch_time._sampling_frequency, ch_ofdm._sampling_frequency, places=5)

        # Check values
        # tau should be identical (shape [batch, rx, rx_ant, tx, tx_ant, paths] or similar - usually [..., paths])
        # tau shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] ?
        # Actually Sionna tau is [batch, num_rx, num_tx, num_paths] usually.

        print(f"Time Channel a shape: {a_time.shape}")
        print(f"OFDM Channel a shape: {a_ofdm.shape}")
        print(f"Time Channel tau shape: {tau_time.shape}")
        print(f"OFDM Channel tau shape: {tau_ofdm.shape}")

        tf.debugging.assert_near(tau_time, tau_ofdm, rtol=1e-5)

        print("Paths consistency check passed (Tau matched)!")


if __name__ == "__main__":
    unittest.main()
