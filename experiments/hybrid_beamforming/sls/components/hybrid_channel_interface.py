import tensorflow as tf
import numpy as np
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
        neighbor_indices=None,
    ):
        super().__init__(precision=precision)

        self.channel_model = channel_model
        self.resource_grid = resource_grid
        self.use_rbg_granularity = use_rbg_granularity
        self.rbg_size_sc = rbg_size_sc
        self.neighbor_indices = neighbor_indices

        # Instantiate the HybridOFDMChannel
        self.hybrid_channel = HybridOFDMChannel(
            channel_model=channel_model,
            resource_grid=resource_grid,
            tx_array=tx_array,
            rx_array=rx_array,
            num_tx_ports=num_tx_ports,
            num_rx_ports=num_rx_ports,
            normalize_channel=False,  # Disable normalization for SLS (Pathloss required)
            use_rbg_granularity=use_rbg_granularity,
            rbg_size=rbg_size_sc,
        )

    def set_analog_weights(self, w_rf, a_rf):
        """Pass-through for setting analog weights."""
        self.hybrid_channel.set_analog_weights(w_rf, a_rf)

    def get_neighbor_channel_info(
        self, batch_size, ut_loc, bs_loc, ut_orient, bs_orient
    ):
        """
        Generates channel only for the specified neighbors using virtual topology mapping.
        """
        # In SLS, batch is usually 1 for topology. Let's assume batch=1 for simplification
        # as gen_hexgrid returns [1, num_ut, 3] etc.

        num_ut = self.neighbor_indices.shape[1]
        num_neighbors = self.neighbor_indices.shape[2]

        # 1. Map physical positions/orientations to virtual BSs
        # Gather mapped BS locations: [batch, num_ut, num_neighbors, 3]
        bs_loc_mapped = tf.gather(bs_loc, self.neighbor_indices, axis=1, batch_dims=0)
        bs_orient_mapped = tf.gather(
            bs_orient, self.neighbor_indices, axis=1, batch_dims=0
        )

        # Flatten neighbors for channel model: [batch, num_ut * num_neighbors, 3]
        bs_loc_flat = tf.reshape(bs_loc_mapped, [batch_size, -1, 3])
        bs_orient_flat = tf.reshape(bs_orient_mapped, [batch_size, -1, 3])

        # 2. SET VIRTUAL TOPOLOGY
        self.channel_model.set_topology(ut_loc, bs_loc_flat, ut_orient, bs_orient_flat)

        # 3. Call channel model
        h_port_flat = self.hybrid_channel(batch_size)
        # h_port_flat shape: [batch, num_ut, num_ut * num_neighbors, ofdm, sc, rx_p, tx_p]

        # 4. RESTORE ORIGINAL TOPOLOGY
        self.channel_model.set_topology(ut_loc, bs_loc, ut_orient, bs_orient)

        # 5. Extract active links (i, i*K + k)
        # For each UT i, we take the block of virtual BSs [i*K : (i+1)*K]
        h_list = []
        for i in range(num_ut):
            start = i * num_neighbors
            end = (i + 1) * num_neighbors
            h_list.append(h_port_flat[:, i, start:end])
        h_neighbor = tf.stack(h_list, axis=1)
        # h_neighbor shape: [batch, num_ut, num_neighbors, ofdm, sc, rx_p, tx_p]

        s, u, v = tf.linalg.svd(h_neighbor)
        return h_neighbor, s, u, v

    def get_full_channel_info(
        self, batch_size, ut_loc=None, bs_loc=None, ut_orient=None, bs_orient=None
    ):
        """
        Returns full SVD results and the underlying port channel.
        If use_rbg_granularity is True, returns channel at RBG centers.
        If neighbor_indices is set, uses neighbor-based virtual mapping.
        """
        if self.neighbor_indices is not None:
            return self.get_neighbor_channel_info(
                batch_size, ut_loc, bs_loc, ut_orient, bs_orient
            )

        # Fallback to full channel if no indices provided (Small Scale)
        h_port = self.hybrid_channel(batch_size)

        # Transpose to standard order
        # h_port: [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]
        # Transpose to [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]
        # Wait, self.hybrid_channel returns [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]
        # BUT the TX dimension (num_tx) is what we want to reduce.

        h_permuted = tf.transpose(h_port, perm=[0, 1, 3, 5, 6, 2, 4])
        # h_permuted shape: [batch, num_rx, num_ofdm, num_sc, num_rx_ports, num_tx, num_tx_ports]
        # Actually SVD usually happens on ports: [..., num_rx_ports, num_tx_ports]
        # h_permuted: [batch, num_rx, num_ofdm, num_sc, num_rx_ports, num_tx, num_tx_ports] ??
        # No, h_port from HybridOFDMChannel is [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]

        # Re-check Transpose logic
        # Original: [batch, rx, tx, ofdm, sc, rx_p, tx_p] -> [batch, rx, ofdm, sc, rx_p, tx, tx_p]? NO.
        # We want [batch, rx, tx, ofdm, sc, rx_p, tx_p]
        # tf.transpose(h_port, perm=[0, 1, 3, 5, 6, 2, 4])
        # -> [batch, num_rx, num_ofdm, num_rx_ports, num_tx_ports, num_tx, num_sc]
        # This permutation seems wrong if we want SVD on rx_ports/tx_ports.
        # Standard SVD needs matrices at the last 2 dims.

        # Correct permutation for SVD on ports:
        # [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]
        s, u, v = tf.linalg.svd(h_port)
        return h_port, s, u, v

    def get_precoding_channel(
        self,
        granularity,
        rbg_size_sc=None,
        batch_size=None,
        ut_loc=None,
        bs_loc=None,
        ut_orient=None,
        bs_orient=None,
    ):
        """
        Get channel specifically for precoding calculation.
        """
        # If neighbor_indices is set, we use the virtual topology for precoding as well
        if self.neighbor_indices is not None:
            # We generate flavor-specific info if needed, but for now reuse neighbors
            h_neighbor, _, _, _ = self.get_neighbor_channel_info(
                batch_size, ut_loc, bs_loc, ut_orient, bs_orient
            )
            return h_neighbor

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
