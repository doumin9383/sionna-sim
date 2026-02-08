import tensorflow as tf
from sionna.phy import Block
from experiments.hybrid_beamforming.shared.channel_models import HybridOFDMChannel


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
        use_rbg_granularity=False,
        rbg_size_sc=1,
    ):
        super().__init__(precision=precision)

        self.channel_model = channel_model
        self.resource_grid = resource_grid
        self.use_rbg_granularity = use_rbg_granularity
        self.rbg_size_sc = rbg_size_sc

        # Instantiate the HybridOFDMChannel
        self.hybrid_channel = HybridOFDMChannel(
            channel_model=channel_model,
            resource_grid=resource_grid,
            tx_array=tx_array,
            rx_array=rx_array,
            num_tx_ports=num_tx_ports,
            num_rx_ports=num_rx_ports,
            normalize_channel=True,  # Ensure consistent normalization
            use_rbg_granularity=use_rbg_granularity,
            rbg_size=rbg_size_sc,
        )

    def set_analog_weights(self, w_rf, a_rf):
        """Pass-through for setting analog weights."""
        self.hybrid_channel.set_analog_weights(w_rf, a_rf)

    def get_full_channel_info(self, batch_size):
        """
        Returns full SVD results and the underlying port channel.
        If use_rbg_granularity is True, returns channel at RBG centers.
        """
        h_port = self.hybrid_channel(batch_size)
        # Permute: [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]
        # Note: if use_rbg_granularity is True, num_sc will be num_rbgs
        h_permuted = tf.transpose(h_port, perm=[0, 1, 3, 5, 6, 2, 4])
        s, u, v = tf.linalg.svd(h_permuted)
        return h_permuted, s, u, v

    def get_precoding_channel(self, batch_size, granularity, rbg_size_sc=None):
        """
        Get channel specifically for precoding calculation.
        """
        if granularity == "Wideband":
            # Treat as 1 huge RBG covering everything
            # Sionna's get_rbg_channel likely needs the size.
            total_sc = self.resource_grid.num_effective_subcarriers
            h_port = self.hybrid_channel.get_rbg_channel(batch_size, rbg_size=total_sc)
        elif granularity == "Subband":
            if rbg_size_sc is None:
                raise ValueError("rbg_size_sc required for Subband")
            h_port = self.hybrid_channel.get_rbg_channel(
                batch_size, rbg_size=rbg_size_sc
            )
        else:  # Narrowband
            h_port = self.hybrid_channel(batch_size)

        # Permute to standard shape for precoding [batch, num_rx, num_tx, num_ofdm, num_freq_blocks, num_rx_ports, num_tx_ports]
        h_permuted = tf.transpose(h_port, perm=[0, 1, 3, 5, 6, 2, 4])

        return h_permuted

    def call(self, batch_size):
        return self.get_full_channel_info(batch_size)
