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

    def get_serving_pathloss(
        self,
        batch_size,
        serving_bs_ids=None,
        ut_loc=None,
        bs_loc=None,
        ut_orient=None,
        bs_orient=None,
        ut_velocities=None,
        in_state=None,
    ):
        """
        Computes pathloss from the serving BS to each UT based on the generated channel.
        Uses sionna.sys.utils.get_pathloss to extract pathloss from channel coefficients.

        Args:
            batch_size (int): Batch size
            serving_bs_ids (tf.Tensor): [batch_size, num_ut] IDs of serving BSs.
                                      If None, assumes Neighbor 0 is always the serving BS.

        Returns:
            tf.Tensor: Pathloss in dB [batch_size, num_ut]
        """
        # 1. External Loader Case
        if self.external_loader is not None:
            # If external loader has explicit power map, use it
            if (
                hasattr(self.external_loader, "get_power_map")
                and self.hybrid_channel._global_topology
            ):
                pass

        # Use explicitly provided topology or fallback to global topology
        ut_loc = (
            ut_loc
            if ut_loc is not None
            else self.hybrid_channel._global_topology["ut_loc"]
        )
        bs_loc = (
            bs_loc
            if bs_loc is not None
            else self.hybrid_channel._global_topology["bs_loc"]
        )
        ut_orient = (
            ut_orient
            if ut_orient is not None
            else self.hybrid_channel._global_topology["ut_orient"]
        )
        bs_orient = (
            bs_orient
            if bs_orient is not None
            else self.hybrid_channel._global_topology["bs_orient"]
        )
        ut_velocities = (
            ut_velocities
            if ut_velocities is not None
            else self.hybrid_channel._global_topology["ut_velocities"]
        )
        in_state = (
            in_state
            if in_state is not None
            else self.hybrid_channel._global_topology["in_state"]
        )

        if self.neighbor_indices is None:
            raise ValueError(
                "neighbor_indices must be set to calculate serving pathloss"
            )

        # Slice neighbor_indices for Serving BS (index 0)
        serving_neighbor_indices = self.neighbor_indices[:, :, 0:1]  # [B, U, 1]

        h_serving = self.get_neighbor_channel_info(
            batch_size,
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orient=ut_orient,
            bs_orient=bs_orient,
            neighbor_indices=serving_neighbor_indices,
            ut_velocities=ut_velocities,
            in_state=in_state,
            return_element_channel=True,
            return_s_u_v=False,
        )

        # h_serving: [B, U, 1, RxAnt, TxAnt, Time, Freq]
        # remove neighbor dim
        h_serving = tf.squeeze(h_serving, axis=2)  # [B, U, RxA, TxA, T, F]

        # Use simple averaging to get pathloss (linear)
        # PL_lin = 1 / E[|h|^2]
        # sionna.sys.utils.get_pathloss expects [..., Rx, RxAnt, Tx, TxAnt, T, F]
        # But here dimensions are [B, U(Rx), RxA, TxA, T, F] (Tx is implicit/single serving)

        # Manual calculation to be safe and avoid dimension shuffling for utility
        rx_power = tf.reduce_mean(
            tf.square(tf.abs(h_serving)), axis=[-1, -2, -3, -4]
        )  # Mean over F, T, TxA, RxA

        # Avoid zero division
        rx_power = tf.maximum(rx_power, 1e-20)
        pl_lin = 1.0 / rx_power

        pl_db = 10.0 * tf.math.log(pl_lin) / tf.math.log(10.0)

        return pl_db

    def call(self, batch_size):
        """Mandatory block entry point."""
        return self.hybrid_channel(batch_size)
