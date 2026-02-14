import tensorflow as tf
import numpy as np
from sionna.phy import Block
from wsim.common.channel.hybrid import GenerateHybridBeamformingOFDMChannel


class HybridChannelInterface(Block):
    """
    Interface for Hybrid Beamforming Channel.
    Focused on scenario control and interference optimization.
    Physical computations are delegated to GenerateHybridBeamformingOFDMChannel.
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
        external_loader=None,
    ):
        super().__init__(precision=precision)

        self.channel_model = channel_model
        self.resource_grid = resource_grid
        self.use_rbg_granularity = use_rbg_granularity
        self.rbg_size_sc = rbg_size_sc
        self.neighbor_indices = neighbor_indices
        self.external_loader = external_loader

        # Instantiate the GenerateHybridBeamformingOFDMChannel
        self.hybrid_channel = GenerateHybridBeamformingOFDMChannel(
            channel_model=channel_model,
            resource_grid=resource_grid,
            tx_array=tx_array,
            rx_array=rx_array,
            num_tx_ports=num_tx_ports,
            num_rx_ports=num_rx_ports,
            normalize_channel=False,  # Disable normalization for SLS (Pathloss required)
        )

    def set_analog_weights(self, w_rf, a_rf):
        """Pass-through for setting analog weights."""
        self.hybrid_channel.set_analog_weights(w_rf, a_rf)

    def get_neighbor_channel_info(
        self,
        batch_size,
        ut_loc,
        bs_loc,
        ut_orient,
        bs_orient,
        neighbor_indices=None,
        ut_velocities=None,
        in_state=None,
        return_element_channel=False,
        return_s_u_v=True,
    ):
        """
        Requests port-domain channel from Base class using ID-Based Sparse Calculation.
        """
        current_neighbor_indices = (
            neighbor_indices if neighbor_indices is not None else self.neighbor_indices
        )
        if current_neighbor_indices is None:
            raise ValueError("neighbor_indices is required.")

        # 1. Update Base Class Topology (for Statistical Models)
        if self.external_loader is None:
            self.hybrid_channel.set_topology(
                ut_loc=ut_loc,
                bs_loc=bs_loc,
                ut_orient=ut_orient,
                bs_orient=bs_orient,
                ut_velocities=ut_velocities,
                in_state=in_state,
                store=True,
            )

        # 2. Call Base Class ID-Based Calculation
        # Returns [Batch, UT, Neighbor, RxPort, TxPort, Time, Freq]
        h_channel = self.hybrid_channel.compute_specific_links(
            batch_size=batch_size,
            neighbor_indices=current_neighbor_indices,
            external_loader=self.external_loader,
            return_element_channel=return_element_channel,
            chunk_size=36,
            ut_orient=ut_orient,
            bs_orient=bs_orient,
            ut_velocities=ut_velocities,
        )

        # frequency resolution alignment (The "Expand" Fix)
        # If use_rbg_granularity is True, return channel sampled at RBG centers
        if self.use_rbg_granularity:
            # h_channel: [Batch, UT, Neighbor, RxP, TxP, Time, Freq]
            # Freq is the last dimension
            h_channel = h_channel[..., :: self.rbg_size_sc]

        if return_element_channel or not return_s_u_v:
            return h_channel

        # 3. Compute SVD for Digital Beamforming (Return S, U, V)
        # tf.linalg.svd expects [..., M, N]. h_channel: [B, U, N, RP, TP, T, F]
        # Transpose to [B, U, N, T, F, RP, TP] for SVD if needed, but wait:
        # The simulator expects [B, U, N, F, RP, TP] usually for processing.
        # Actually, get_neighbor_channel_info returns [B, U, N, RP, TP, T, F]
        s, u, v = tf.linalg.svd(h_channel)
        return h_channel, s, u, v

    def get_element_channel_for_beam_selection(
        self,
        batch_size,
        ut_loc,
        bs_loc,
        ut_orient,
        bs_orient,
        neighbor_indices=None,
        ut_velocities=None,
        in_state=None,
    ):
        """
        Shorthand for obtaining element-domain channel.
        """
        return self.get_neighbor_channel_info(
            batch_size=batch_size,
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orient=ut_orient,
            bs_orient=bs_orient,
            neighbor_indices=neighbor_indices,
            ut_velocities=ut_velocities,
            in_state=in_state,
            return_element_channel=True,
            return_s_u_v=False,
        )

    def approximate_distant_interference(
        self, ut_loc, bs_loc, neighbor_indices_to_mask, p_tx_watt=0.251
    ):
        """
        Phase 3.6: Calculate approximate interference from non-neighbor BSs.
        Computes distance-based Path Loss and returns estimated interference power [Watt].
        """
        # (Simplified distance-based model)
        # ut_loc: [Batch, UT, 3], bs_loc: [Batch, BS, 3]
        # 1. Distances
        diff = tf.expand_dims(ut_loc, 2) - tf.expand_dims(bs_loc, 1)  # [B, U, BS, 3]
        dist_3d = tf.norm(diff, axis=-1)  # [B, U, BS]

        # 2. Simplified Path Loss (3GPP UMi style logic or generic)
        # pl = 32.4 + 21*log10(d) + 20*log10(fc)
        # Using a very simple alpha=3.5 exponent for placeholder
        # p_rx = p_tx / (dist^alpha)
        int_power = p_tx_watt / tf.pow(tf.maximum(dist_3d, 1.0), 3.5)

        # 3. Mask neighbors
        # neighbor_indices_to_mask: [Batch, UT, NeighborCount]
        # Create a mask for all BSs
        num_bs = tf.shape(bs_loc)[1]
        bs_ids = tf.range(num_bs)
        # mask is [Batch, UT, BS]
        mask = tf.reduce_any(
            tf.equal(
                tf.expand_dims(tf.expand_dims(bs_ids, 0), 0),
                tf.expand_dims(neighbor_indices_to_mask, -1),
            ),
            axis=2,
        )

        int_power_masked = tf.where(mask, 0.0, int_power)

        return tf.reduce_sum(int_power_masked, axis=2, keepdims=True)  # [B, U, 1]

    def call(self, batch_size):
        """Mandatory block entry point."""
        return self.hybrid_channel(batch_size)
