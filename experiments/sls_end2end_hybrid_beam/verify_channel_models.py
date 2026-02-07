import tensorflow as tf
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel.tr38901 import PanelArray
from experiments.sls_end2end_hybrid_beam.components.channel_models import (
    RBGChannelModel,
    ChunkedTimeChannel,
    HybridOFDMChannel,
)


def test_rbg_channel_model():
    print("Testing RBGChannelModel...")

    # Setup arrays
    bs_array = PanelArray(
        num_rows_per_panel=4,
        num_cols_per_panel=4,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="38.901",
        carrier_frequency=3.5e9,
    )

    ut_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=3.5e9,
    )

    # Instantiate RBGChannelModel
    # This should not fail due to invalid arguments passed to super().__init__
    channel_model = RBGChannelModel(
        scenario="umi",
        carrier_frequency=3.5e9,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
    )

    # Set dummy topology
    batch_size = 2
    num_ut = 1
    channel_model.set_topology(
        ut_loc=tf.zeros((batch_size, num_ut, 3)),
        bs_loc=tf.zeros((batch_size, 1, 3)),
        ut_orientations=tf.zeros((batch_size, num_ut, 3)),
        bs_orientations=tf.zeros((batch_size, 1, 3)),
        ut_velocities=tf.zeros((batch_size, num_ut, 3)),
        in_state=tf.zeros((batch_size, num_ut), dtype=tf.bool),
    )

    print("RBGChannelModel instantiated and topology set successfully.")
    return channel_model, bs_array, ut_array


def test_chunked_time_channel(channel_model):
    print("\nTesting ChunkedTimeChannel...")
    bandwidth = 10e6
    num_time_samples = 1000
    l_min, l_max = -6, 20

    time_channel = ChunkedTimeChannel(
        channel_model=channel_model,
        bandwidth=bandwidth,
        num_time_samples=num_time_samples,
        l_min=l_min,
        l_max=l_max,
    )

    h_time = time_channel.get_cir(batch_size=2)
    print(f"h_time shape: {h_time.shape}")
    # Expected: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths (actually num_taps), num_time_samples?]
    # No, cir_to_time_channel returns:
    # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
    # Check rank
    assert len(h_time.shape) == 7
    print("ChunkedTimeChannel.get_cir() passed.")


def test_hybrid_ofdm_channel(channel_model, bs_array, ut_array):
    print("\nTesting HybridOFDMChannel...")

    # Resource Grid
    rg = ResourceGrid(num_ofdm_symbols=14, fft_size=512, subcarrier_spacing=30e3)

    num_tx_ports = 4
    num_rx_ports = 2

    hybrid_channel = HybridOFDMChannel(
        channel_model=channel_model,
        resource_grid=rg,
        tx_array=bs_array,
        rx_array=ut_array,
        num_tx_ports=num_tx_ports,
        num_rx_ports=num_rx_ports,
    )

    # Test RBG Channel
    rbg_size = 16
    h_rbg = hybrid_channel.get_rbg_channel(batch_size=2, rbg_size=rbg_size)
    print(f"h_rbg shape: {h_rbg.shape}")

    expected_rbgs = 512 // 16
    # cir_to_ofdm_channel output: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps(num_syms), num_freqs]
    assert h_rbg.shape[-1] == expected_rbgs
    print(f"RBG Channel correctly returned {expected_rbgs} frequency points.")

    # Test Hybrid BF (Full Call)
    h_port = hybrid_channel(batch_size=2)
    print(f"h_port shape: {h_port.shape}")

    # Expected: [batch, num_rx, num_rx_port, num_tx, num_tx_port, num_syms, num_sc]
    # num_tx_port = 4, num_rx_port = 2
    assert h_port.shape[2] == num_rx_ports  # rx_port is at index 2 (brqtpsc)
    assert h_port.shape[4] == num_tx_ports  # tx_port is at index 4

    print("HybridOFDMChannel passed.")


if __name__ == "__main__":
    try:
        model, bs, ut = test_rbg_channel_model()
        test_chunked_time_channel(model)
        test_hybrid_ofdm_channel(model, bs, ut)
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback

        traceback.print_exc()
