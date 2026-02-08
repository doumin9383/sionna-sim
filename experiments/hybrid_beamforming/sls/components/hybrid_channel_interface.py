import tensorflow as tf
import numpy as np
from sionna.phy import Block, PI
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
        # 5. Extract active links
        h_list = []

        # Check if Uplink (Rx=BS, Shape=[Batch, NumVirtualBS, NumUT, ...])
        # or Downlink (Rx=UT, Shape=[Batch, NumUT, NumVirtualBS, ...])
        # num_virtual_bs = num_ut * num_neighbors
        is_uplink = h_port_flat.shape[1] == (num_ut * num_neighbors)

        for i in range(num_ut):
            start = i * num_neighbors
            end = (i + 1) * num_neighbors

            if is_uplink:
                # Uplink: [B, BS(Rx), RxP, UT(Tx), TxP, S, C]
                # Slice BS=start:end, UT=i
                # Result: [B, Neighbors, RxP, TxP, S, C]
                chan_slice = h_port_flat[:, start:end, :, i, ...]
            else:
                # Downlink: [B, UT(Rx), RxP, BS(Tx), TxP, S, C]
                # Slice UT=i, BS=start:end
                # Result: [B, RxP, Neighbors, TxP, S, C]
                chan_slice = h_port_flat[:, i, :, start:end, ...]
                # Transpose to [B, Neighbors, RxP, TxP, S, C]
                chan_slice = tf.transpose(chan_slice, [0, 2, 1, 3, 4, 5])

            h_list.append(chan_slice)

        h_neighbor = tf.stack(h_list, axis=1)

        s, u, v = tf.linalg.svd(h_neighbor)
        return h_neighbor, s, u, v

    def _get_steering_vector(self, array, theta, phi):
        """
        Manually computes 3GPP-compliant steering vector for a PanelArray.
        theta: [..., L] Zenith angle [0, pi]
        phi: [..., L] Azimuth angle [-pi, pi]
        Returns: [..., L, num_ant] complex response
        """
        # 1. Gain Pattern (Element Field)
        f_pol1_theta, f_pol1_phi = array.ant_pol1.field(theta, phi)
        e_gain = tf.sqrt(tf.abs(f_pol1_theta) ** 2 + tf.abs(f_pol1_phi) ** 2)

        # 2. Array Phase Shift
        pos = array.ant_pos  # [N, 3]

        k_y = tf.sin(theta) * tf.sin(phi)
        k_z = tf.cos(theta)

        # [..., L, 1] * [1, N] -> [..., L, N]
        # array.ant_pos is already in meters.
        # Phase = 2*pi * (k . pos) / lambda
        phase = (
            2.0
            * PI
            * (
                tf.expand_dims(k_y, -1) * tf.expand_dims(pos[:, 1], 0)
                + tf.expand_dims(k_z, -1) * tf.expand_dims(pos[:, 2], 0)
            )
            / array._lambda_0
        )

        e_array = tf.complex(tf.cos(phase), tf.sin(phase))

        # Apply element gain [..., L, N]
        return tf.cast(e_gain, e_array.dtype)[..., tf.newaxis] * e_array

    def get_external_neighbor_channel_info(
        self, batch_size, ut_loc, bs_loc, ut_orient, bs_orient
    ):
        """
        Generates channel using external ray-tracing data (Zarr).
        Efficient MIMO recovery via direct contraction.
        """
        num_rx_ports = self.hybrid_channel.num_rx_ports
        num_tx_ports = self.hybrid_channel.num_tx_ports

        # 1. Find nearest mesh for each UT
        ut_mesh_indices = self.external_loader.find_nearest_mesh(ut_loc)

        # 2. Get paths (a, tau, angles) from Zarr for (UT, NeighborBS) pairs
        a, tau, doa_az, doa_el, dod_az, dod_el = self.external_loader.get_paths(
            batch_size, self.neighbor_indices, ut_mesh_indices
        )

        # 3. Calculate Port-Domain Path Gains (a') using Steering Vectors and Analog Weights
        # RX Steering Vectors: [B, U, K, L, RX_ANT]
        e_rx = self._get_steering_vector(self.hybrid_channel.rx_array, doa_el, doa_az)

        # TX Steering Vectors: [B, U, K, L, TX_ANT]
        e_tx = self._get_steering_vector(self.hybrid_channel.tx_array, dod_el, dod_az)

        # アナログ重み
        w_rf = self.hybrid_channel.w_rf
        a_rf = self.hybrid_channel.a_rf

        # RX Weight Contraction (v_rx = A_rf^H * e_rx)
        # a_rf: [RX_A, RX_P], [B, RX_A, RX_P], [B, U, RX_A, RX_P]
        if len(a_rf.shape) == 2:  # [j, r]
            v_rx = tf.einsum("jr,buklj->buklr", tf.math.conj(a_rf), e_rx)
        elif len(a_rf.shape) == 3:  # [b, j, r]
            v_rx = tf.einsum("bjr,buklj->buklr", tf.math.conj(a_rf), e_rx)
        else:  # [b, u, j, r]
            v_rx = tf.einsum("bujr,buklj->buklr", tf.math.conj(a_rf), e_rx)

        # TX Weight Contraction (v_tx = e_tx^H * W_rf)
        # w_rf: [TX_A, TX_P], [B, TX_A, TX_P], [B, S, TX_A, TX_P]
        if len(w_rf.shape) == 2:  # [j, p]
            v_tx = tf.einsum("buklj,jp->buklp", tf.math.conj(e_tx), w_rf)
        elif len(w_rf.shape) == 3:  # [b, j, p]
            v_tx = tf.einsum("buklj,bjp->buklp", tf.math.conj(e_tx), w_rf)
        else:  # [b, s, j, p]
            w_rf_neighbors = tf.gather(
                w_rf, self.neighbor_indices, axis=1, batch_dims=0
            )
            # w_rf_neighbors: [b, u, k, j, p]
            v_tx = tf.einsum("buklj,bukjp->buklp", tf.math.conj(e_tx), w_rf_neighbors)

        # a_port = a * v_rx * v_tx: [B, U, K, L, RX_P, TX_P]
        # v_rx: [B, U, K, L, RX_P], v_tx: [B, U, K, L, TX_P]
        a_sq = tf.squeeze(a, axis=[2, 4, 6])
        a_port = tf.einsum("bukl,buklr,buklp->buklrp", a_sq, v_rx, v_tx)

        # 4. Convert CIR to OFDM Response (Port Domain)
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

        # a_port shape for cir_to_ofdm_channel: [B, num_ut, num_rx_ports, num_neighbors, num_tx_ports, L, num_time_steps]
        # Our a_port: [B, U, K, L, RX_P, TX_P]
        # Transpose to: [B, U, RX_P, K, TX_P, L, 1]
        a_port_sionna = tf.transpose(a_port, [0, 1, 4, 2, 5, 3])[..., tf.newaxis]

        # tau shape: [B, U, K, L] -> [B, U, RX_P, K, TX_P, L] is broadcasted by cir_to_ofdm_channel if rank 4
        # Wait, if rank 4, it's [B, RX, TX, L].
        # In our case it's [B, U, K, L].
        # For cir_to_ofdm_channel to work with neighbor mapping, we need to match the rank-4 logic or pad.
        # Actually, let's just make tau match the rank-6 shape [B, U, RX_P, K, TX_P, L]
        tau_sionna = tf.tile(
            tau[:, :, tf.newaxis, :, tf.newaxis, :],
            [1, 1, num_rx_ports, 1, num_tx_ports, 1],
        )

        # h_port: [B, U, RX_P, K, TX_P, 1, SC]
        h_port = cir_to_ofdm_channel(
            frequencies, a_port_sionna, tau_sionna, normalize=False
        )

        # Reshape to simulator expected shape: [B, U, K, 1, SC, RX_P, TX_P]
        # h_port current: [B, U, RX_P, K, TX_P, 1, SC]
        h_neighbor = tf.transpose(h_port, [0, 1, 3, 5, 6, 2, 4])

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
