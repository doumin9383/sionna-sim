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
        external_loader=None,
    ):
        super().__init__(precision=precision)

        self.channel_model = channel_model
        self.resource_grid = resource_grid
        self.use_rbg_granularity = use_rbg_granularity
        self.rbg_size_sc = rbg_size_sc
        self.neighbor_indices = neighbor_indices
        self.external_loader = external_loader

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

    def call(self, batch_size):
        """Mandatory call method for Block."""
        return self.get_full_channel_info(batch_size)

    def set_analog_weights(self, w_rf, a_rf):
        """Pass-through for setting analog weights."""
        self.hybrid_channel.set_analog_weights(w_rf, a_rf)

    def get_neighbor_channel_info(
        self, batch_size, ut_loc, bs_loc, ut_orient, bs_orient
    ):
        """
        Generates channel only for the specified neighbors using virtual topology mapping.
        """
        num_ut = self.neighbor_indices.shape[1]
        num_neighbors = self.neighbor_indices.shape[2]

        # 1. Map physical positions/orientations to virtual BSs
        bs_loc_mapped = tf.gather(bs_loc, self.neighbor_indices, axis=1, batch_dims=0)
        bs_orient_mapped = tf.gather(
            bs_orient, self.neighbor_indices, axis=1, batch_dims=0
        )

        # Flatten neighbors for channel model
        bs_loc_flat = tf.reshape(bs_loc_mapped, [batch_size, -1, 3])
        bs_orient_flat = tf.reshape(bs_orient_mapped, [batch_size, -1, 3])

        # 2. SET VIRTUAL TOPOLOGY
        self.channel_model.set_topology(ut_loc, bs_loc_flat, ut_orient, bs_orient_flat)

        # 3. Call channel model
        h_port_flat = self.hybrid_channel(batch_size)

        # 4. RESTORE ORIGINAL TOPOLOGY
        self.channel_model.set_topology(ut_loc, bs_loc, ut_orient, bs_orient)

        # 5. Extract active links
        h_list = []
        for i in range(num_ut):
            start = i * num_neighbors
            end = (i + 1) * num_neighbors
            h_list.append(h_port_flat[:, i, start:end])
        h_neighbor = tf.stack(h_list, axis=1)

        s, u, v = tf.linalg.svd(h_neighbor)
        return h_neighbor, s, u, v

    def get_external_neighbor_channel_info(
        self, batch_size, ut_loc, bs_loc, ut_orient, bs_orient
    ):
        """
        Generates channel using external ray-tracing data (Zarr).
        Bypasses TR38901 to avoid OOM.
        """
        # 1. Find nearest mesh for each UT
        ut_mesh_indices = self.external_loader.find_nearest_mesh(ut_loc)

        # 2. Get paths (a, tau) from Zarr for (UT, NeighborBS) pairs
        a, tau = self.external_loader.get_paths(
            batch_size, self.neighbor_indices, ut_mesh_indices
        )

        # 3. Convert CIR to OFDM Response
        if self.use_rbg_granularity:
            from experiments.hybrid_beamforming.shared.channel_models import (
                subcarrier_frequencies,
            )

            num_subcarriers = self.resource_grid.fft_size
            num_rbgs = tf.maximum(num_subcarriers // self.rbg_size_sc, 1)
            rbg_indices = tf.range(num_rbgs) * self.rbg_size_sc + (
                self.rbg_size_sc // 2
            )
            all_frequencies = subcarrier_frequencies(
                num_subcarriers, self.resource_grid.subcarrier_spacing
            )
            frequencies = tf.gather(all_frequencies, rbg_indices)
        else:
            frequencies = self.resource_grid.frequencies

        from sionna.phy.channel import cir_to_ofdm_channel

        # h_link: [batch, num_ut, num_neighbors, num_ofdm, num_sc]
        h_link = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)

        # 4. Reconstruct Port Domain Channel
        # For Phase 3 MVP, we broadcast scalar results to MIMO dimensions.
        num_rx_ports = self.hybrid_channel.num_rx_ports
        num_tx_ports = self.hybrid_channel.num_tx_ports

        h_neighbor = tf.expand_dims(tf.expand_dims(h_link, axis=-1), axis=-1)
        h_neighbor = tf.tile(h_neighbor, [1, 1, 1, 1, 1, num_rx_ports, num_tx_ports])
        h_neighbor = h_neighbor / tf.sqrt(
            tf.cast(num_rx_ports * num_tx_ports, h_neighbor.dtype)
        )

        s, u, v = tf.linalg.svd(h_neighbor)
        return h_neighbor, s, u, v

    def get_full_channel_info(
        self, batch_size, ut_loc=None, bs_loc=None, ut_orient=None, bs_orient=None
    ):
        """
        Returns the port-domain channel information.
        If use_rbg_granularity is True, returns channel at RBG centers.
        If neighbor_indices is set, uses neighbor-based virtual mapping.
        """
        if self.external_loader is not None and self.neighbor_indices is not None:
            return self.get_external_neighbor_channel_info(
                batch_size, ut_loc, bs_loc, ut_orient, bs_orient
            )
        if self.neighbor_indices is not None:
            return self.get_neighbor_channel_info(
                batch_size, ut_loc, bs_loc, ut_orient, bs_orient
            )

        if self.use_rbg_granularity:
            h = self.hybrid_channel.get_rbg_channel(batch_size, self.rbg_size_sc)
        else:
            h = self.hybrid_channel(batch_size)

        s, u, v = tf.linalg.svd(h)
        return h, s, u, v

    def get_precoding_channel(
        self, granularity, rbg_size_sc, batch_size, ut_loc, bs_loc, ut_orient, bs_orient
    ):
        """
        Returns the channel matrix to be used for precoding calculation.
        """
        if self.external_loader is not None and self.neighbor_indices is not None:
            return self.get_external_neighbor_channel_info(
                batch_size, ut_loc, bs_loc, ut_orient, bs_orient
            )[0]

        return self.get_full_channel_info(
            batch_size, ut_loc, bs_loc, ut_orient, bs_orient
        )[0]
