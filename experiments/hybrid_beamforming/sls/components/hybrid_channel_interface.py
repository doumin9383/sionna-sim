import tensorflow as tf
import numpy as np
from sionna.phy import Block, PI
from .hybrid_channels import GenerateHybridBeamformingOFDMChannel


class HybridChannelInterface(Block):
    """
    Interface for Hybrid Beamforming Channel.
    Wraps GenerateHybridBeamformingOFDMChannel to provide effective channel gains via SVD.
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

    def call(self, batch_size):
        """Mandatory call method for Block."""
        return self.get_full_channel_info(batch_size)

    def set_analog_weights(self, w_rf, a_rf):
        """Pass-through for setting analog weights."""
        self.global_w_rf = w_rf
        self.global_a_rf = a_rf
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
        rbg_size_sc=1,
        return_s_u_v=True,
    ):
        """
        Generates channel only for the specified neighbors using ID-Based Sparse Calculation.
        Delegates all physical channel generation to Base class.
        """
        current_neighbor_indices = (
            neighbor_indices if neighbor_indices is not None else self.neighbor_indices
        )
        if current_neighbor_indices is None:
            raise ValueError(
                "neighbor_indices must be provided either in init or in method call."
            )

        # 1. Update Base Class Topology Storage (for Statistical Models)
        # If external loader is used, set_topology is bypassed in Base, but we pass info via kwargs.
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

        # 2. Prepare Topology Kwargs (for External Path)
        topology_kwargs = {
            "ut_loc": ut_loc,
            "bs_loc": bs_loc,
            "ut_orient": ut_orient,
            "bs_orient": bs_orient,
            "ut_velocities": ut_velocities,
            "in_state": in_state,
        }

        # 3. Call Base Class ID-Based Calculation
        h_channel = self.hybrid_channel.compute_specific_links(
            batch_size=batch_size,
            neighbor_indices=current_neighbor_indices,
            external_loader=self.external_loader,
            return_element_channel=return_element_channel,
            # Use smaller chunk_size to avoid OOM.
            # rbg_size_sc might be large (Wideband), so we don't use it for chunking memory.
            chunk_size=36,
            **topology_kwargs,
        )

        # 4. Return Results (with SVD if requested)
        if return_element_channel:
            return h_channel

        if not return_s_u_v:
            return h_channel

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
        rbg_size_sc=1,
    ):
        """
        Get element-domain channel for beam selection.
        Delegates to get_neighbor_channel_info with return_element_channel=True.
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
            rbg_size_sc=rbg_size_sc,
            return_s_u_v=False,
        )

    def _compute_approximate_interference(self, ut_loc, bs_loc, neighbor_indices):
        """
        Stub for Phase 3.6: Calculate approximate interference from non-neighbor BSs.
        Computes distance-based Path Loss and returns estimated interference power.
        """
        # Placeholder implementation
        # 1. Calculate distances to all BSs
        # 2. Mask neighbors
        # 3. Apply simplified Path Loss model
        # 4. Sum power
        return tf.zeros([tf.shape(ut_loc)[0], tf.shape(ut_loc)[1], 1])  # [Batch, UT, 1]

    def get_full_channel_info(
        self,
        batch_size,
        ut_loc=None,
        bs_loc=None,
        ut_orient=None,
        bs_orient=None,
        neighbor_indices=None,
        ut_velocities=None,
        in_state=None,
        return_s_u_v=True,
    ):
        """
        Returns the port-domain channel information.
        If use_rbg_granularity is True, returns channel at RBG centers.
        If neighbor_indices is set, uses neighbor-based virtual mapping.
        """
        # Prioritize passed neighbor_indices, fall back to self.neighbor_indices
        effective_neighbor_indices = (
            neighbor_indices if neighbor_indices is not None else self.neighbor_indices
        )

        if effective_neighbor_indices is not None:
            return self.get_neighbor_channel_info(
                batch_size,
                ut_loc,
                bs_loc,
                ut_orient,
                bs_orient,
                neighbor_indices=effective_neighbor_indices,
                ut_velocities=ut_velocities,
                in_state=in_state,
                return_s_u_v=return_s_u_v,
            )

        if self.use_rbg_granularity:
            h = self.hybrid_channel.get_rbg_channel(batch_size, self.rbg_size_sc)
        else:
            h = self.hybrid_channel(batch_size)

        if not return_s_u_v:
            return h

        s, u, v = tf.linalg.svd(h)
        return h, s, u, v
