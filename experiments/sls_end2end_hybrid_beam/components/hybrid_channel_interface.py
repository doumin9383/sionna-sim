import tensorflow as tf
from sionna.phy import Block
from .hybrid_channels import HybridOFDMChannel


class HybridChannelInterface(Block):
    """
    Interface for Hybrid Beamforming Channel.
    Wraps HybridOFDMChannel to provide effective channel gains via SVD.
    """

    def __init__(
        self,
        channel_model,
        resource_grid,
        tx_array,
        rx_array,
        num_tx_ports,
        num_rx_ports,
        precision=None,
    ):
        super().__init__(precision=precision)

        self.channel_model = channel_model
        self.resource_grid = resource_grid

        # Instantiate the HybridOFDMChannel
        self.hybrid_channel = HybridOFDMChannel(
            channel_model=channel_model,
            resource_grid=resource_grid,
            tx_array=tx_array,
            rx_array=rx_array,
            num_tx_ports=num_tx_ports,
            num_rx_ports=num_rx_ports,
            normalize_channel=True,  # Ensure consistent normalization
        )

    def set_analog_weights(self, w_rf, a_rf):
        """Pass-through for setting analog weights."""
        self.hybrid_channel.set_analog_weights(w_rf, a_rf)

    def get_full_channel_info(self, batch_size):
        """
        Returns full SVD results and the underlying port channel.
        """
        h_port = self.hybrid_channel.get_port_channel(batch_size)
        # Permute: [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]
        h_permutes = tf.transpose(h_port, perm=[0, 1, 3, 5, 6, 2, 4])
        s, u, v = tf.linalg.svd(h_permutes)
        return h_permutes, s, u, v

    def call(self, batch_size):
        return self.get_full_channel_info(batch_size)
