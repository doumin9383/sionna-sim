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
        # Batch processing for UTs to avoid OOM
        # Process UTs in small batches (e.g., 10)
        batch_size_ut = 10
        h_chunks = []

        # We process all UTs
        # neighbor_indices: [batch, num_ut, num_neighbors]
        # We need to loop over the 'num_ut' dimension

        num_ut_total = self.neighbor_indices.shape[1]
        num_neighbors = self.neighbor_indices.shape[2]

        # Determine strict loop count
        num_batches_ut = (num_ut_total + batch_size_ut - 1) // batch_size_ut

        for i in range(num_batches_ut):
            start_ut = i * batch_size_ut
            end_ut = min(start_ut + batch_size_ut, num_ut_total)
            current_batch_size = end_ut - start_ut

            # Slicing inputs for current UT batch
            # ut_loc: [batch, num_ut, 3] -> [batch, current_batch_size, 3]
            ut_loc_batch = ut_loc[:, start_ut:end_ut, :]
            ut_orient_batch = ut_orient[:, start_ut:end_ut, :]

            # Additional states that need slicing if they exist in channel model
            # in_state is usually [batch, num_ut]
            # ut_velocities is [batch, num_ut, 3]
            # We access them from channel_model properties if not passed explicitly,
            # but set_topology expects arguments if we want to update them.
            # Ideally `get_neighbor_channel_info` should receive them or we extract from model.
            # But `ut_loc` etc are passed as args.

            # Let's check `channel_model` properties for current state
            # SystemLevelChannel wraps vector scenario in `_scenario`
            if hasattr(self.channel_model, "indoor"):
                in_state_full = self.channel_model.indoor
                ut_vel_full = self.channel_model.ut_velocities
            else:
                # Fallback for Sionna < 0.19 or internal structure
                in_state_full = self.channel_model._scenario.indoor
                ut_vel_full = self.channel_model._scenario.ut_velocities

            in_state_batch = in_state_full[:, start_ut:end_ut]
            ut_vel_batch = ut_vel_full[:, start_ut:end_ut, :]

            # neighbor_indices for this batch: [batch, current_batch_size, num_neighbors]
            neighbor_indices_batch = self.neighbor_indices[:, start_ut:end_ut, :]

            # 1. Map physical positions/orientations to virtual BSs for this batch
            # bs_loc: [batch, num_bs, 3]
            # Gather BSs relevant to these UTs
            bs_loc_mapped = tf.gather(
                bs_loc, neighbor_indices_batch, axis=1, batch_dims=0
            )
            bs_orient_mapped = tf.gather(
                bs_orient, neighbor_indices_batch, axis=1, batch_dims=0
            )

            # Flatten neighbors for channel model: [batch, current_batch_size * num_neighbors, 3]
            bs_loc_flat = tf.reshape(bs_loc_mapped, [batch_size, -1, 3])
            bs_orient_flat = tf.reshape(bs_orient_mapped, [batch_size, -1, 3])

            # 2. SET VIRTUAL TOPOLOGY (Small batch)
            self.channel_model.set_topology(
                ut_loc=ut_loc_batch,
                bs_loc=bs_loc_flat,
                ut_orientations=ut_orient_batch,
                bs_orientations=bs_orient_flat,
                ut_velocities=ut_vel_batch,
                in_state=in_state_batch,
            )

            # 3. Call channel model
            # This generates H for all pairs in the batch:
            # (current_batch_size UTs) x (current_batch_size * num_neighbors Virtual BSs)
            # Total links generated: current_batch_size * (current_batch_size * num_neighbors)
            # However, we only care about the diagonal blocks (UT i <-> Neighbors of UT i)
            # But Sionna generates full mesh.
            # Batch size = 10 -> 10 * 80 = 800 links. Much smaller than 150 * 1200 = 180,000.
            h_port_flat = self.hybrid_channel(batch_size)

            # 4. Extract active links
            # h_port_flat shape depends on direction.
            # Uplink: [Batch, NumVirtualBS, RxP, NumUT, TxP, S, C]
            # Downlink: [Batch, NumUT, RxP, NumVirtualBS, TxP, S, C]

            # NumVirtualBS in this batch
            num_virtual_bs_batch = current_batch_size * num_neighbors
            is_uplink = h_port_flat.shape[1] == num_virtual_bs_batch

            h_list_batch = []

            for k in range(current_batch_size):
                # Map k to flattened structure
                # Virtual BSs for k-th UT in batch are at indices [k*num_neighbors : (k+1)*num_neighbors]
                start_idx = k * num_neighbors
                end_idx = (k + 1) * num_neighbors

                if is_uplink:
                    # Rx=VirtualBS (BS), Tx=UT
                    # Slice VirtualBS=start:end, UT=k
                    # [Batch, Neighbors, RxP, TxP, S, C]
                    # Note: h_port_flat indices: [B, VirtualBS, RxP, UT, TxP, S, C]
                    chan_slice = h_port_flat[:, start_idx:end_idx, :, k, ...]
                else:
                    # Rx=UT, Tx=VirtualBS
                    # Slice UT=k, VirtualBS=start:end
                    # [Batch, RxP, Neighbors, TxP, S, C]
                    chan_slice = h_port_flat[:, k, :, start_idx:end_idx, ...]
                    # Transpose to [B, Neighbors, RxP, TxP, S, C]
                    chan_slice = tf.transpose(chan_slice, [0, 2, 1, 3, 4, 5])

                h_list_batch.append(chan_slice)

            # Stack batch results: [Batch, current_batch_size, Neighbors, RxP, TxP, S, C]
            h_chunk = tf.stack(h_list_batch, axis=1)
            h_chunks.append(h_chunk)

            # 5. CLEAR TOPOLOGY (Implicitly handled by next set_topology, but good practice if explicitly needed)

        # 4. RESTORE ORIGINAL TOPOLOGY (Ideally, but usually Simulator calls set_topology every step)
        # We should restore it to avoid side effects if 'get_neighbor_channel_info' is called
        # but then 'channel_model' is used elsewhere without setting topology.
        # However, for pure specific use, it might be fine.
        # To be safe, let's restore.
        self.channel_model.set_topology(ut_loc, bs_loc, ut_orient, bs_orient)

        # 6. Concatenate all chunks
        # [Batch, Total_UT, Neighbors, RxP, TxP, S, C]
        h_neighbor = tf.concat(h_chunks, axis=1)

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
