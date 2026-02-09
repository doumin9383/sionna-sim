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
    ):
        """
        Generates channel only for the specified neighbors using virtual topology mapping.
        """
        current_neighbor_indices = (
            neighbor_indices if neighbor_indices is not None else self.neighbor_indices
        )
        if current_neighbor_indices is None:
            raise ValueError(
                "neighbor_indices must be provided either in init or in method call."
            )
        # Batch processing for UTs to avoid OOM
        # Process UTs in small batches (e.g., 1)
        batch_size_ut = 1
        h_chunks = []

        # We process all UTs
        # neighbor_indices: [batch, num_ut, num_neighbors]
        # We need to loop over the 'num_ut' dimension

        num_ut_total = current_neighbor_indices.shape[1]
        num_neighbors = current_neighbor_indices.shape[2]

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

            # Prepare additional states
            if ut_velocities is not None:
                ut_vel_full = ut_velocities
            elif hasattr(self.channel_model, "ut_velocities"):
                ut_vel_full = self.channel_model.ut_velocities
            elif hasattr(self.channel_model, "_scenario") and hasattr(
                self.channel_model._scenario, "ut_velocities"
            ):
                ut_vel_full = self.channel_model._scenario.ut_velocities
            else:
                # Default to zeros if not found [batch, num_ut, 3]
                ut_vel_full = tf.zeros_like(ut_loc)

            if in_state is not None:
                in_state_full = in_state
            elif hasattr(self.channel_model, "indoor"):
                in_state_full = self.channel_model.indoor
            elif hasattr(self.channel_model, "_scenario") and hasattr(
                self.channel_model._scenario, "indoor"
            ):
                in_state_full = self.channel_model._scenario.indoor
            else:
                # Default to false (outdoor) [batch, num_ut]
                in_state_full = tf.zeros(ut_loc.shape[:2], dtype=tf.bool)

            in_state_batch = in_state_full[:, start_ut:end_ut]
            ut_vel_batch = ut_vel_full[:, start_ut:end_ut, :]

            # neighbor_indices for this batch: [batch, current_batch_size, num_neighbors]
            neighbor_indices_batch = current_neighbor_indices[:, start_ut:end_ut, :]

            # 1. Map physical positions/orientations to virtual BSs for this batch
            # bs_loc: [batch, num_bs, 3]
            # Gather BSs relevant to these UTs
            # Use batch_dims=1 to gather per-batch-item
            bs_loc_mapped = tf.gather(
                bs_loc, neighbor_indices_batch, axis=1, batch_dims=1
            )
            bs_orient_mapped = tf.gather(
                bs_orient, neighbor_indices_batch, axis=1, batch_dims=1
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
            if return_element_channel:
                # Element Domain Channel (No Analog Weights)
                # Use get_element_rbg_channel explicitly
                h_port_flat = self.hybrid_channel.get_element_rbg_channel(
                    batch_size, rbg_size=rbg_size_sc
                )
            else:
                # Port Domain Channel (With Analog Weights)
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

        # Transpose to [Batch, Total_UT, Neighbors, S, C, RxP, TxP]
        # to match simulator expectations [Batch, U, Neighbors, Time, SC, RxP, TxP]
        # S=Time, C=Subcarrier (from Sionna GenerateOFDMChannel convention [..., Time, Subcarrier])
        # Indices: 0=Batch, 1=U, 2=Neighbors, 3=RxP, 4=TxP, 5=Time, 6=SC
        # Target: 0, 1, 2, 5, 6, 3, 4
        h_neighbor = tf.transpose(h_neighbor, perm=[0, 1, 2, 5, 6, 3, 4])

        if return_element_channel:
            return h_neighbor

        s, u, v = tf.linalg.svd(h_neighbor)
        return h_neighbor, s, u, v

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
        Get element-domain channel for beam selection, handling topology and neighbors.
        """
        effective_neighbor_indices = (
            neighbor_indices if neighbor_indices is not None else self.neighbor_indices
        )

        # Check external loader
        if self.external_loader is not None:
            return self.get_external_neighbor_channel_info(
                batch_size,
                ut_loc,
                bs_loc,
                ut_orient,
                bs_orient,
                neighbor_indices=effective_neighbor_indices,
                return_element_channel=True,
            )

        # Re-use get_neighbor_channel_info with flag
        return self.get_neighbor_channel_info(
            batch_size,
            ut_loc,
            bs_loc,
            ut_orient,
            bs_orient,
            neighbor_indices=effective_neighbor_indices,
            ut_velocities=ut_velocities,
            in_state=in_state,
            return_element_channel=True,
            rbg_size_sc=rbg_size_sc,
        )

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
        self,
        batch_size,
        ut_loc,
        bs_loc,
        ut_orient,
        bs_orient,
        neighbor_indices=None,
        return_element_channel=False,
    ):
        """
        Generates channel using external ray-tracing data (Zarr).
        Reconstructs channel using stored Rays and LSPs.
        """
        current_neighbor_indices = (
            neighbor_indices if neighbor_indices is not None else self.neighbor_indices
        )
        if current_neighbor_indices is None:
            raise ValueError(
                "neighbor_indices must be provided either in init or in method call."
            )

        # Imports to avoid circular dependency
        from sionna.phy.channel.tr38901 import Rays, Topology
        from sionna.phy.channel import cir_to_ofdm_channel
        from experiments.hybrid_beamforming.shared.channel_models import (
            subcarrier_frequencies,
        )

        num_rx_ports = self.hybrid_channel.num_rx_ports
        num_tx_ports = self.hybrid_channel.num_tx_ports

        # Batch processing for UTs to avoid OOM
        batch_size_ut = 4  # Adjust based on memory
        h_chunks = []

        num_ut_total = current_neighbor_indices.shape[1]
        num_neighbors = current_neighbor_indices.shape[2]

        # Prepare frequencies for OFDM conversion
        num_subcarriers = self.resource_grid.fft_size
        if self.use_rbg_granularity:
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

        # Loop over UT batches
        num_batches_ut = (num_ut_total + batch_size_ut - 1) // batch_size_ut

        for i in range(num_batches_ut):
            start_ut = i * batch_size_ut
            end_ut = min(start_ut + batch_size_ut, num_ut_total)
            current_batch_size = end_ut - start_ut

            # Indices for slicing
            # ut_indices: [start_ut ... end_ut]
            ut_indices = tf.range(start_ut, end_ut)

            # neighbor_indices_batch: [B, SubUT, Neighbors]
            neighbor_indices_batch = current_neighbor_indices[:, start_ut:end_ut, :]

            # Slicing Inputs
            # ut_loc_batch: [Batch, SubUT, 3]
            ut_loc_batch = ut_loc[:, start_ut:end_ut, :]
            ut_orient_batch = ut_orient[:, start_ut:end_ut, :]

            # 1. Retrieve Rays & LSPs for this batch
            data = self.external_loader.get_rays(ut_indices=ut_indices, bs_indices=None)

            # Helper to gather neighbors
            def gather_neighbors(tensor, indices, axis, batch_dims):
                return tf.gather(tensor, indices, axis=axis, batch_dims=batch_dims)

            # Unpack Rays Data [B, U, BS, P] -> Gather neighbors
            delays = gather_neighbors(
                data["delays"], neighbor_indices_batch, axis=2, batch_dims=2
            )
            powers = gather_neighbors(
                data["powers"], neighbor_indices_batch, axis=2, batch_dims=2
            )
            aoa = gather_neighbors(
                data["aoa"], neighbor_indices_batch, axis=2, batch_dims=2
            )
            aod = gather_neighbors(
                data["aod"], neighbor_indices_batch, axis=2, batch_dims=2
            )
            zoa = gather_neighbors(
                data["zoa"], neighbor_indices_batch, axis=2, batch_dims=2
            )
            zod = gather_neighbors(
                data["zod"], neighbor_indices_batch, axis=2, batch_dims=2
            )
            xpr = gather_neighbors(
                data["xpr"], neighbor_indices_batch, axis=2, batch_dims=2
            )

            # Unpack LSPs [B, BS, U] -> Transpose -> Gather
            pathloss = tf.transpose(data["pathloss"], [0, 2, 1])  # [B, U, BS]
            shadow_fading = tf.transpose(data["shadow_fading"], [0, 2, 1])
            k_factor = tf.transpose(data["k_factor"], [0, 2, 1])

            pl_batch = gather_neighbors(
                pathloss, neighbor_indices_batch, axis=2, batch_dims=2
            )
            sf_batch = gather_neighbors(
                shadow_fading, neighbor_indices_batch, axis=2, batch_dims=2
            )
            k_batch = gather_neighbors(
                k_factor, neighbor_indices_batch, axis=2, batch_dims=2
            )

            # 2. Flatten to Links for _cir_sampler [TotalLinks]
            total_links = current_batch_size * num_neighbors

            def flatten_to_links(tensor):
                shape = tf.shape(tensor)
                new_shape = tf.concat(
                    [[batch_size * total_links], shape[3:]], axis=0
                )  # [TotalLinks, ...]
                flat = tf.reshape(tensor, new_shape)
                return flat[:, tf.newaxis, tf.newaxis, ...]

            rays_obj_flat = Rays(
                delays=flatten_to_links(delays),
                powers=flatten_to_links(powers),
                aoa=flatten_to_links(aoa),
                aod=flatten_to_links(aod),
                zoa=flatten_to_links(zoa),
                zod=flatten_to_links(zod),
                xpr=flatten_to_links(xpr),
            )

            # Flatten LSPs [TotalLinks, 1, 1] (Ensure these are defined!)
            k_flat = flatten_to_links(k_batch[..., tf.newaxis])[..., 0]
            sf_flat = flatten_to_links(sf_batch[..., tf.newaxis])[..., 0]
            pl_flat = flatten_to_links(pl_batch[..., tf.newaxis])[..., 0]

            # Flatten LSPs [TotalLinks, 1, 1]
            # We gather axis 1.
            # gather(bs_loc, indices, axis=1, batch_dims=1)
            # bs_loc needs to be broadcast to [Batch, SubUT, AllBS, 3]?
            # No, bs_loc is [Batch, AllBS, 3].
            # neighbor_indices is [Batch, SubUT, Neighbors].
            # This gathering is tricky.
            # Use gather_nd? Or broadcast bs_loc?
            # bs_loc_exp = tf.expand_dims(bs_loc, 1) # [B, 1, AllBS, 3]
            # bs_loc_tiled = tf.tile(bs_loc_exp, [1, current_batch_size, 1, 1]) # [B, U, AllBS, 3]
            # Then gather axis 2.
            # Optimized:
            # Flatten indices [B*U*N]. Flatten bs_loc [B*AllBS].
            # But B=1 usually.

            # Simple gather logic:
            # bs_loc_mapped = tf.gather(bs_loc, neighbor_indices_batch, axis=1, batch_dims=1)
            # This works if neighbor_indices_batch refers to AllBS index. Yes it does.
            bs_loc_mapped = tf.gather(
                bs_loc, neighbor_indices_batch, axis=1, batch_dims=1
            )  # [B, U, N, 3]

            ut_loc_expanded = tf.repeat(
                ut_loc_batch, repeats=num_neighbors, axis=2
            )  # [B, U*N?, 3]? No.
            ut_loc_tiled = tf.tile(
                tf.expand_dims(ut_loc_batch, 2), [1, 1, num_neighbors, 1]
            )  # [B, U, N, 3]
            ut_orient_tiled = tf.tile(
                tf.expand_dims(ut_orient_batch, 2), [1, 1, num_neighbors, 1]
            )

            # Flatten Locations
            ut_loc_flat = tf.reshape(
                ut_loc_tiled, [-1, 1, 3]
            )  # [TotalLinks, 1, 3] (Rx)
            bs_loc_flat = tf.reshape(
                bs_loc_mapped, [-1, 1, 3]
            )  # [TotalLinks, 1, 3] (Tx)
            ut_orient_flat = tf.reshape(ut_orient_tiled, [-1, 1, 3])
            # BS orientations need gathering too
            bs_orient_mapped = tf.gather(
                bs_orient, neighbor_indices_batch, axis=1, batch_dims=1
            )
            bs_orient_flat = tf.reshape(bs_orient_mapped, [-1, 1, 3])

            # Construct dummy Topology
            # We provide minimal info needed by _step_11
            # _step_11 needs orientations and velocities (for Doppler).
            # We assume velocity is 0 if not provided or handle it.
            # Construct dummy Topology with all required fields
            # We provide minimal info needed by _step_11 (Doppler & Orientations)

            ut_vel = tf.zeros_like(ut_loc_flat)

            # Shapes: [Batch, NumBS, NumUT] -> [TotalLinks, 1, 1]
            dummy_zeros = tf.zeros([batch_size * total_links, 1, 1], dtype=tf.float32)
            dummy_los = tf.zeros([batch_size * total_links, 1, 1], dtype=tf.bool)

            topo = Topology(
                velocities=ut_vel,  # [TotalLinks, 1, 3]
                moving_end="rx",
                los_aoa=dummy_zeros,
                los_aod=dummy_zeros,
                los_zoa=dummy_zeros,
                los_zod=dummy_zeros,
                los=dummy_los,
                distance_3d=dummy_zeros,
                tx_orientations=bs_orient_flat,
                rx_orientations=ut_orient_flat,
            )

            # Retrieve c_ds from scenario [Batch, AllUT, NumBS]
            # c_ds is [Batch, Rx, Tx]. For DL, Rx=UT. So [Batch, AllUT, NumBS].
            c_ds_full = self.channel_model._scenario.get_param("cDS") * 1e-9

            # 1. Slice for current UT batch [B, SubUT, NumBS]
            c_ds_batch_ut = tf.gather(c_ds_full, ut_indices, axis=1)

            # 2. Gather neighbors [B, SubUT, Neighbors]
            c_ds_batch = gather_neighbors(
                c_ds_batch_ut, neighbor_indices_batch, axis=2, batch_dims=2
            )

            c_ds_flat = flatten_to_links(c_ds_batch[..., tf.newaxis])[..., 0]

            c_ds = c_ds_flat  # [TotalLinks, 1, 1]

            # Call Sampler
            # h comes out as [TotalLinks, Rx(1), RxAnt, Tx(1), TxAnt, Paths, Time]
            h_element, delays_flat = self.channel_model._cir_sampler(
                1,  # num_time_samples
                30e3,  # sampling_frequency (dummy)
                k_flat,
                rays_obj_flat,
                topo,
                c_ds,
            )

            # 5. Apply Pathloss and Shadow Fading
            # gain = 10^(-PL/20) * sqrt(SF)
            # pl_flat [Links, 1, 1], sf_flat [Links, 1, 1]
            gain_lin = tf.sqrt(sf_flat) * tf.pow(10.0, -pl_flat / 20.0)
            h_element = (
                h_element
                * tf.complex(gain_lin, 0.0)[
                    :, :, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis
                ]
            )

            # Sum over paths (Step 11 usually returns paths? No, Step 11 returns Multi-path components?
            # ChannelCoefficientsGenerator returns [..., num_paths, num_time_steps].
            # h_element is [Links, 1, RxA, 1, TxA, Paths, Time].
            # We need to apply Doppler? Step 11 includes Doppler.
            # We need to sum over paths to get CIR?
            # cir_to_ofdm_channel takes 'a' (paths) and 'tau'.
            # It expects [..., Paths, Time].
            # So we KEEP paths distinct for OFDM conversion!
            # h_element IS 'a' (complex gains per path).

            # 6. Apply Analog Weights (Contraction)
            # Observed h_element shape: [Links, Rx(1), Tx(1), Paths, TxAnt, RxAnt, Time]
            # We need [Links, RxAnt, TxAnt, Paths, 1]

            # Remove singleton Rx/Tx dims (indices 1 and 2)
            h_e = tf.squeeze(
                h_element, axis=[1, 2]
            )  # [Links, Paths, TxAnt, RxAnt, Time]

            # Transpose to [Links, RxAnt, TxAnt, Paths, Time]
            # Input determined to be [Links, Paths, RxAnt, TxAnt, Time] based on tracebacks.
            # Permutation: 0(Links), 2(RxAnt), 3(TxAnt), 1(Paths), 4(Time)
            h_e = tf.transpose(
                h_e, perm=[0, 2, 3, 1, 4]
            )  # [Links, RxAnt, TxAnt, Paths, Time]

            if return_element_channel:
                # User requested Element Channel (e.g. for Beam Selection)
                # Convert to OFDM
                # h_e: [Links, RxA, TxA, Paths, 1]
                # tau_flat: [Links, 1, 1, Paths] (Need to broadcast to RxA/TxA? cir_to_ofdm handles broadcasting if ranks align)
                # h_e has rank 5. tau_flat has rank 4.
                # cir_to_ofdm(frequencies, a, tau). 'a' and 'tau' shape tail should match.
                # 'tau' should be [..., Paths]. 'a' is [..., Paths, Time].

                # Flatten dimensions to avoid Rank > 5 error in cir_to_ofdm_channel on GPU
                # h_e: [Links, RxA, TxA, Paths, 1]
                shape_h = tf.shape(h_e)
                num_links = shape_h[0]
                rx_ant = shape_h[1]
                tx_ant = shape_h[2]
                num_paths_dim = shape_h[3]

                # Broadcast tau to match h_e dims
                # Use delays_flat returned from _cir_sampler (includes sub-clustering)
                tau_flat = delays_flat
                # tau_flat shape likely [Links, 1, 1, Paths] (or similar rank 4).

                tau_broadcast = tf.broadcast_to(
                    tau_flat, [num_links, rx_ant, tx_ant, num_paths_dim]
                )

                # Flatten to [N, Paths, 1] and [N, Paths]
                h_e_reshaped = tf.reshape(h_e, [-1, num_paths_dim, 1])
                tau_reshaped = tf.reshape(tau_broadcast, [-1, num_paths_dim])

                # Manual CIR to OFDM (Avoiding Rank>5 and hardcoded axis errors)
                from sionna.phy import PI

                # tau: [N, P]. freq: [F].
                # We need [N, P, F]
                phase = (
                    -2
                    * PI
                    * tau_reshaped[..., tf.newaxis]
                    * frequencies[tf.newaxis, tf.newaxis, :]
                )
                basis = tf.exp(tf.complex(0.0, phase))  # [N, P, F]

                # h_e: [N, P, 1]
                h_term = (
                    tf.complex(h_e_reshaped, 0.0)
                    if h_e_reshaped.dtype.is_floating
                    else h_e_reshaped
                )

                # Multiply and Sum over paths
                h_ofdm_flat = tf.reduce_sum(h_term * basis, axis=1)  # [N, F]

                # Reshape back to [Links, RxA, TxA, 1, SC]
                # h_ofdm_flat is [N, SC]. Need to add Time dim?
                # Target: [num_links, rx_ant, tx_ant, 1, SC]
                h_ofdm = tf.reshape(h_ofdm_flat, [num_links, rx_ant, tx_ant, 1, -1])

                # Squeeze Time (dim 3)
                h_ofdm_squeezed = tf.squeeze(h_ofdm, axis=3)  # [Links, Rx, Tx, SC]

                # Reshape back to [Batch, SubUT, Neighbors, RxA, TxA, SC]
                h_chunk_shape = [
                    batch_size,
                    current_batch_size,
                    num_neighbors,
                    self.hybrid_channel.rx_array.num_ant,
                    self.hybrid_channel.tx_array.num_ant,
                    -1,
                ]
                h_chunk = tf.reshape(h_ofdm_squeezed, h_chunk_shape)

                # Output shape: [B, U, N, Rx, Tx, SC]
                # Matches BeamSelector expectation [..., Rx, Tx, SC] (Rx=2, Tx=32)

                h_chunks.append(h_chunk)
                continue  # Next batch

                h_chunks.append(h_chunk)
                continue  # Next batch

            # Weights need to be gathered/tiled to [Links, ...]
            # w_rf: [b, s, j, p] or similar.
            w_rf_full = self.hybrid_channel.w_rf
            a_rf_full = self.hybrid_channel.a_rf

            # Handle w_rf shape
            neighbor_indices_flat = tf.reshape(neighbor_indices_batch, [-1])

            # Gather w_rf
            if len(w_rf_full.shape) == 4:  # [B, S, Ant, Port]
                w_rf_links = tf.gather(
                    w_rf_full[0], neighbor_indices_flat
                )  # Assume B=1
            elif len(w_rf_full.shape) == 2:  # [Ant, Port] (Global)
                w_rf_links = tf.expand_dims(w_rf_full, 0)
                w_rf_links = tf.tile(w_rf_links, [tf.shape(h_e)[0], 1, 1])
            else:
                w_rf_links = tf.gather(w_rf_full, neighbor_indices_flat)

            # Gather a_rf (Rx)
            # a_rf [Batch, UT, Ant, Port] or [Ant, Port]
            ut_indices_repeated = tf.repeat(ut_indices, num_neighbors)

            if len(a_rf_full.shape) == 4:  # [B, U, Ant, Port]
                a_rf_links = tf.gather(a_rf_full[0], ut_indices_repeated)
            elif len(a_rf_full.shape) == 2:  # [Ant, Port] (Global)
                a_rf_links = tf.expand_dims(a_rf_full, 0)
                a_rf_links = tf.tile(a_rf_links, [tf.shape(h_e)[0], 1, 1])
            else:
                a_rf_links = tf.gather(a_rf_full, ut_indices_repeated)

            # Contract
            # h_e: [Links, RxA, TxA, Paths, 1]
            # a_rf_links: [Links, RxA, RxP]
            # w_rf_links: [Links, TxA, TxP]

            # v_rx = a^H * h
            # einsum("lrp, lrtpx -> lptpx", conj(a_rf), h_e)
            term1 = tf.einsum("lrp, lrtki -> lptki", tf.math.conj(a_rf_links), h_e)

            # v_tx = term1 * w
            # einsum("lptki, lto -> lpok i", term1, w_rf) -> [Links, RxP, Paths, TxP, 1] ?
            # indices: l=Links, p=RxP, t=TxA, k=Paths, i=Time.
            # w: l=Links, t=TxA, o=TxP.
            # result: l, p, o, k, i.
            h_port_links = tf.einsum("lptki, lto -> lpoki", term1, w_rf_links)
            # Shape: [Links, RxP, TxP, Paths, 1]

            # Manual CIR to OFDM for h_port_links (Contracted Channel) to avoid Rank > 5 errors
            # h_port_links: [Links, RxP, TxP, Paths, 1]
            # delays_flat: [Links, 1, 1, Paths] (Need broadcasting)

            shape_t2 = tf.shape(h_port_links)
            # Flatten to [N, Paths, 1]
            num_paths_dim = shape_t2[3]
            h_port_links_reshaped = tf.reshape(h_port_links, [-1, num_paths_dim, 1])

            # Broadcast tau to match h_port_links shape prefix
            # h_port_links shape: [Links, RxP, TxP, Paths, 1]
            # delays_flat shape: [Links, 1, 1, Paths]
            # We want tau to broad cast to [Links, RxP, TxP, Paths]

            # Extract dims
            n_links = shape_t2[0]
            n_rxp = shape_t2[1]
            n_txp = shape_t2[2]

            # Reshape delays_flat to [Links, 1, 1, Paths] (It is already)
            # Tile/Broadcast to [Links, RxP, TxP, Paths]
            tau_broadcast = tf.broadcast_to(
                delays_flat, [n_links, n_rxp, n_txp, num_paths_dim]
            )

            # Flatten tau to [-1, Paths]
            tau_reshaped = tf.reshape(tau_broadcast, [-1, num_paths_dim])

            # Manual DFT
            from sionna.phy import PI

            # phase: [N_flat, Paths, Freq]
            phase = (
                -2
                * PI
                * tau_reshaped[..., tf.newaxis]
                * frequencies[tf.newaxis, tf.newaxis, :]
            )
            basis = tf.exp(tf.complex(0.0, phase))  # [N_flat, P, F]

            term2_c = (
                tf.complex(h_port_links_reshaped, 0.0)
                if h_port_links_reshaped.dtype.is_floating
                else h_port_links_reshaped
            )

            # Sum over paths
            h_ofdm_flat = tf.reduce_sum(term2_c * basis, axis=1)  # [N_flat, F]

            # Reshape back to [Links, RxP, TxP, 1, SC]
            # h_ofdm_flat is [Links*RxP*TxP, SC]
            h_ofdm = tf.reshape(h_ofdm_flat, [n_links, n_rxp, n_txp, 1, -1])

            # 8. Reshape back to [Batch, SubUT, Neighbors, ...]
            # [B, U, N, RxP, TxP, 1, SC]
            h_chunk_shape = [
                batch_size,
                current_batch_size,
                num_neighbors,
            ] + h_ofdm.shape[1:]
            h_chunk = tf.reshape(h_ofdm, h_chunk_shape)

            # Transpose to Simulator format: [B, U, Neighbors, 1(Time), SC, RxP, TxP]
            # Current: [B, U, N, RxP, TxP, 1, SC]
            # Perm: 0, 1, 2, 5, 6, 3, 4
            h_chunk = tf.transpose(h_chunk, perm=[0, 1, 2, 5, 6, 3, 4])

            h_chunks.append(h_chunk)

        # Concatenate all chunks
        h_neighbor = tf.concat(h_chunks, axis=1)

        if batch_size > 1:  # Handle if original batch_size was actually drops?
            # But here we used batch_size=1 inside calculations.
            pass

        s, u, v = tf.linalg.svd(h_neighbor)
        return h_neighbor, s, u, v

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

        if self.external_loader is not None and effective_neighbor_indices is not None:
            # Note: get_external_neighbor_channel_info also needs update to accept neighbor_indices if we want full consistency,
            # but for now let's assume external loader path handles it or we update it too.
            # Let's check get_external_neighbor_channel_info signature.
            # It uses self.neighbor_indices. We should probably update it too or temporarilly set self.neighbor_indices.
            # For safety, let's update it in next step if needed.
            # For now, let's just assume we need to pass it.
            return self.get_external_neighbor_channel_info(
                batch_size,
                ut_loc,
                bs_loc,
                ut_orient,
                bs_orient,
                neighbor_indices=effective_neighbor_indices,
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
            )

        if self.use_rbg_granularity:
            h = self.hybrid_channel.get_rbg_channel(batch_size, self.rbg_size_sc)
        else:
            h = self.hybrid_channel(batch_size)

        s, u, v = tf.linalg.svd(h)
        return h, s, u, v
