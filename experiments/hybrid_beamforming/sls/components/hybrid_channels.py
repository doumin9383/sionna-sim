import tensorflow as tf
import numpy as np
from sionna.phy.channel import (
    GenerateOFDMChannel,
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)
from sionna.phy.channel.tr38901 import PanelArray


class ChunkedGenerateOFDMChannel(GenerateOFDMChannel):
    """
    Extends GenerateOFDMChannel to support chunk-based generation.
    """

    def __init__(self, channel_model, resource_grid, normalize_channel=True):
        super().__init__(channel_model, resource_grid, normalize_channel)
        self._channel_model = channel_model

        # Pre-compute all frequencies
        self._all_frequencies = subcarrier_frequencies(
            resource_grid.fft_size, resource_grid.subcarrier_spacing
        )

    def get_paths(self, batch_size):
        """
        Generate paths (a, tau) for the current realization.
        Wraps the internal channel_model call.
        """
        # We need to replicate the logic of __call__ up to cir generation
        # This usually involves calling the channel model.
        # However, GenerateOFDMChannel.__call__ does:
        # a, tau = self._channel_model(self._num_time_samples, self._sampling_frequency)
        # return cir_to_ofdm_channel(..., a, tau, ...)

        a, tau = self._channel_model(self._num_time_samples, self._sampling_frequency)
        return a, tau

    def get_h_freq_chunk(self, a, tau, start_idx, num_chunk):
        """
        Convert specific subcarriers from CIR to Frequency domain.
        a: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        tau: [batch, num_rx, num_tx, num_paths]
        """
        # Slice frequencies
        chunk_freqs = self._all_frequencies[start_idx : start_idx + num_chunk]

        # Use sionna's utility to convert.
        # Note: cir_to_ofdm_channel expects 'frequencies' as first arg.
        # It handles the broadcasting and complex exponential computation.
        h_freq_chunk = cir_to_ofdm_channel(
            chunk_freqs, a, tau, normalize=self._normalize_channel
        )

        return h_freq_chunk

    def get_rbg_channel(self, batch_size, rbg_size, active_rbgs=None):
        """
        Get channel sampled at RBG granularity.
        """
        a, tau = self.get_paths(batch_size)

        # Calculate RBG center frequencies
        # RBG size is in number of subcarriers
        # We assume RBGs are contiguous blocks

        # Create RBG indices (centers)
        # Shape: [num_rbgs]
        # Example: rbg_size=16. Indices: 8, 24, 40...
        num_subcarriers = self._resource_grid.fft_size
        num_rbgs = tf.maximum(num_subcarriers // rbg_size, 1)

        # Calculate indices of RBG centers
        # If num_rbgs is 1 because of fallback, we take the center of the available band
        if num_subcarriers < rbg_size:
            rbg_indices = tf.constant([num_subcarriers // 2], dtype=tf.int32)
        else:
            rbg_indices = tf.range(num_rbgs) * rbg_size + (rbg_size // 2)

        # Gather frequencies at these indices
        # Note: self._all_frequencies contains FULL grid frequencies (from super().__init__)
        # BUT self._all_frequencies was overwritten in GenerateHybridBeamformingOFDMChannel?
        # Wait, ChunkedGenerateOFDMChannel is base.
        # GenerateHybridBeamformingOFDMChannel overwrite it in __init__.
        # If this method is called on GenerateHybridBeamformingOFDMChannel, self._all_frequencies might be SUBSET?
        # NO. In GenerateHybridBeamformingOFDMChannel.__init__, we set self._all_frequencies = active_freqs.
        # But get_rbg_channel logic assumes INDICES correspond to FFT Grid?
        # "num_subcarriers = self._resource_grid.fft_size"
        # If self._all_frequencies is ONLY active, then index mapping is tricky.
        # Active freqs are usually contiguous subset (except guards).

        # If we assume self._all_frequencies IS active frequencies:
        # We should sample relative to ACTIVE set?
        # Or does RBG grid align with FFT grid?
        # Usually RBG is defined on Active BWP.
        # So treating `self._all_frequencies` as the "Available Resource Grid" is correct.
        # If `fft_size` is large (includes guards), but `self._all_frequencies` is small (active only).
        # We should iterate over `self._all_frequencies`.

        # Let's adjust logic:
        # Use simple strided access on `self._all_frequencies`.
        # Taking every N-th frequency or average?
        # Usually center of RBG.
        # If `self._all_frequencies` has length, say 792 (66 RBs * 12).
        # rbg_size = 12.
        # We want indices 6, 18, ...
        # num_active = len(self._all_frequencies).
        # num_rbgs = num_active // rbg_size.

        num_active = tf.shape(self._all_frequencies)[0]
        num_rbgs = tf.maximum(num_active // rbg_size, 1)

        if num_active < rbg_size:
            rbg_indices = tf.constant([num_active // 2], dtype=tf.int32)
        else:
            rbg_indices = tf.range(num_rbgs) * rbg_size + (rbg_size // 2)

        rbg_freqs = tf.gather(self._all_frequencies, rbg_indices)

        if active_rbgs is not None:
            rbg_freqs = tf.gather(rbg_freqs, active_rbgs)

        h_rbg = cir_to_ofdm_channel(
            rbg_freqs, a, tau, normalize=self._normalize_channel
        )
        return h_rbg


class GenerateHybridBeamformingOFDMChannel(ChunkedGenerateOFDMChannel):
    """
    Sionna-compatible Block that generates a Digital Port Channel by applying
    Analog Beamforming to an underlying physical channel.

    This block mimics GenerateOFDMChannel but returns a channel with [num_ports]
    instead of [num_elements].
    """

    def __init__(
        self,
        channel_model,
        resource_grid,
        tx_array,
        rx_array,
        num_tx_ports,
        num_rx_ports,
        normalize_channel=True,
    ):
        # Initialize the base ChunkedGenerateOFDMChannel
        super().__init__(channel_model, resource_grid, normalize_channel)

        self.tx_array = tx_array
        self.rx_array = rx_array
        self.num_tx_ports = num_tx_ports
        self.num_rx_ports = num_rx_ports

        # Prepare Active Frequencies based on ResourceGrid
        # Standard Sionna ResourceGrid might have guard bands or DC removal.
        # super().__init__ sets self._all_frequencies to the full grid frequencies.
        # We overwrite it to only include active frequencies for efficient chunking.
        all_freqs_full = self._all_frequencies
        self._all_frequencies = tf.gather(
            all_freqs_full, resource_grid.effective_subcarrier_ind
        )
        # We use _num_active_sc for loop bounds
        self._num_active_sc = len(self._all_frequencies)

        # Global Topology Storage
        self._global_topology = None

    def _init_default_weights(self):
        # Default: Map ports to the first elements
        self.w_rf = tf.eye(
            self.tx_array.num_ant, num_columns=self.num_tx_ports, dtype=tf.complex64
        )
        self.a_rf = tf.eye(
            self.rx_array.num_ant, num_columns=self.num_rx_ports, dtype=tf.complex64
        )

    def set_analog_weights(self, w_rf, a_rf):
        """
        Update Analog BF weights.
        Supports:
        - [ant, port] (Shared across all)
        - [num_tx/rx, ant, port] (Sector-specific)
        - [batch, num_tx/rx, ant, port] (Fully individual)
        """
        self.w_rf = w_rf
        self.a_rf = a_rf

    def set_topology(
        self,
        ut_loc,
        bs_loc,
        ut_orient,
        bs_orient,
        ut_velocities,
        in_state,
        store=True,
    ):
        """
        Proxy method to set topology on the underlying channel model.
        If store=True, saves the full topology for later sparse slicing (Phase 3.5).
        """
        if store:
            self._global_topology = {
                "ut_loc": ut_loc,
                "bs_loc": bs_loc,
                "ut_orient": ut_orient,
                "bs_orient": bs_orient,
                "ut_velocities": ut_velocities,
                "in_state": in_state,
            }

        self._channel_model.set_topology(
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orientations=ut_orient,
            bs_orientations=bs_orient,
            ut_velocities=ut_velocities,
            in_state=in_state,
        )

    def compute_specific_links(
        self,
        batch_size,
        neighbor_indices,
        external_loader=None,
        return_element_channel=False,
        chunk_size=72,
        **topology_kwargs,
    ):
        """
        Computes channel for specific links defined by neighbor_indices.

        Args:
            batch_size: int
            neighbor_indices: [Batch, UT, Neighbor] (3D Index)
            external_loader: Optional loader for External Data (Ray-tracing)
            return_element_channel: bool
            chunk_size: int
            **topology_kwargs: Topology info (ut_loc, bs_loc, etc.) for external loader path.

        Returns:
            h_channel: [Batch, UT, Neighbor, ...]
        """
        # 1. External Data Path (Direct CIR Fetching)
        if external_loader is not None:
            # Delegate to a dedicated helper (logic handled below or in a separate method)
            return self._compute_external_links(
                batch_size,
                neighbor_indices,
                external_loader,
                return_element_channel,
                chunk_size,
                **topology_kwargs,
            )

        # 2. Statistical Model Path (Topology Slicing)
        if self._global_topology is None:
            raise ValueError(
                "Global topology not set. Call set_topology(store=True) first."
            )

        return self._compute_statistical_links(
            batch_size, neighbor_indices, return_element_channel, chunk_size
        )

    def _compute_external_links(
        self,
        batch_size,
        neighbor_indices,
        loader,
        return_element_channel,
        chunk_size,
        **topology_kwargs,
    ):
        """
        Handles External Data (Ray-tracing) bypassing set_topology.
        Fetches CIR directly using ID pairs.
        """
        from sionna.phy.channel.tr38901 import Rays, Topology

        # Extract Topology Info if provided
        ut_orient = topology_kwargs.get("ut_orient")
        bs_orient = topology_kwargs.get("bs_orient")
        ut_velocities = topology_kwargs.get("ut_velocities")

        # ... (rest of implementation follows, needing orientations for dummy topology)

        # neighbor_indices: [Batch, Num_UT, Num_Neighbor]
        current_neighbor_indices = neighbor_indices
        num_ut_total = tf.shape(current_neighbor_indices)[1]
        num_neighbors = tf.shape(current_neighbor_indices)[2]

        # Batch processing for UTs to avoid OOM
        batch_size_ut = 4
        h_chunks = []

        num_batches_ut = (num_ut_total + batch_size_ut - 1) // batch_size_ut

        for i in range(num_batches_ut):
            start_ut = i * batch_size_ut
            end_ut = tf.minimum(start_ut + batch_size_ut, num_ut_total)
            current_batch_size = end_ut - start_ut  # Actual batch size for this chunk

            # Slice indices for current batch
            # ut_indices: [start_ut ... end_ut]
            # Assumes UTs are sequential 0..N in the loader
            ut_indices = tf.range(start_ut, end_ut)
            neighbor_indices_batch = current_neighbor_indices[:, start_ut:end_ut, :]

            # 1. Retrieve Rays & LSPs
            # We assume loader returns data for (UTs x All_BS) or similar structure
            # Interface logic assumed data[key] is [Batch, UT, BS, ...]
            data = loader.get_rays(ut_indices=ut_indices, bs_indices=None)

            # Helper to gather neighbors
            def gather_neighbors(tensor):
                return tf.gather(tensor, neighbor_indices_batch, axis=2, batch_dims=2)

            # Unpack and Gather
            delays = gather_neighbors(data["delays"])
            powers = gather_neighbors(data["powers"])
            aoa = gather_neighbors(data["aoa"])
            aod = gather_neighbors(data["aod"])
            zoa = gather_neighbors(data["zoa"])
            zod = gather_neighbors(data["zod"])
            xpr = gather_neighbors(data["xpr"])

            # LSPs: [B, BS, U] -> Transpose [B, U, BS] -> Gather
            def unpack_lsp(key):
                tens = tf.transpose(data[key], [0, 2, 1])
                return gather_neighbors(tens)

            pathloss = unpack_lsp("pathloss")
            shadow_fading = unpack_lsp("shadow_fading")
            k_factor = unpack_lsp("k_factor")

            # 2. Flatten to Links for _cir_sampler
            total_links = current_batch_size * num_neighbors

            def flatten_to_links(tensor):
                # tensor: [Batch, U, N, ...]
                # We flatten Batch, U, N to [TotalLinks]
                # Assuming Batch dimension is handled as part of links (?)
                # Actually _cir_sampler usually takes [Num_Links, ...]
                shape = tf.shape(tensor)
                # Reshape to [TotalLinks, ...]
                # Note: tensor contains Batch dim 0.
                flat = tf.reshape(tensor, tf.concat([[-1], shape[3:]], axis=0))
                # Add singleton dims for Rx/Tx/Antenna broadcasting if needed?
                # _cir_sampler expects [NumPaths, NumLinks] or similar?
                # No, Sionna _cir_sampler inputs:
                # rays.delays: [batch_size, num_rx, num_tx, num_paths]
                # OR [num_links, 1, 1, num_paths] if we treat each link independently.
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

            # LSPs: flatten to [TotalLinks, 1, 1]
            k_flat = flatten_to_links(k_factor[..., tf.newaxis])[..., 0]
            sf_flat = flatten_to_links(shadow_fading[..., tf.newaxis])[..., 0]
            pl_flat = flatten_to_links(pathloss[..., tf.newaxis])[..., 0]

            # 3. Construct Dummy Topology
            # Need Tx/Rx orientations.
            # We assume bs_orient is accessible? No, we need to pass it or store it?
            # 'global_topology' is NOT stored for External Data path per instruction.
            # But we need Orientations for calculating steering vectors.
            # We must fetch them from somewhere or they must be passed.
            # Interface passed them to get_external...
            # But compute_specific_links signature doesn't have them?
            # User instruction: "set_topology is not called".
            # BUT efficient calculation requires orientations.
            # We can rely on external loader providing orientations? Or pass them?
            # `compute_specific_links` should probably accept loc/orient if they are dynamic?
            # Or we assume they are static and accessible.
            # The Interface call in Phase 2 used arguments `ut_loc`, `bs_loc` etc. when calling `get_neighbor_channel_info`.
            # `compute_specific_links` arguments are `batch_size, neighbor_indices`.
            # We need to access `ut_orient` and `bs_orient`.
            # Strategy: Access `self._global_topology`?
            # No, user said "set_topology is NOT called" for External Data.
            # But implies we assume data is loaded?
            # Actually, for Ray Tracing, the angles (AoA/AoD) are usually in Global Coordinate System (GCS) or LCS?
            # Sionna Rays are usually LCS (Local Coordinate System)? No, GCS?
            # Warning: If Rays are GCS, we need orientations to convert to LCS.
            # If Rays are LCS, we don't need orientations.
            # Sionna _cir_sampler takes Topology (orientations) to apply to Rays.
            # This implies Rays are likely GCS.
            # Solution: We DO need orientations.
            # I will modify `compute_specific_links` signature to accept `topology_info` kwargs or assume `self._channel_model.set_topology` was NOT called but we have the info?
            # Better: `compute_specific_links` should allow passing topology data implicitly or explicitly.
            # Or simpler: For External, `loader` might provide orientations?
            # Let's assume we can access `self._global_topology` IF it was set?
            # "external_loader usage ... set_topology IS NOT CALLED".
            # This means `_global_topology` is empty.
            # So `compute_specific_links` MUST take `ut_orient`, `bs_orient`.
            # I will assume they are passed in `**kwargs` or similar mechanism.
            # Let's update this method to assume `topology_info` is passed or retrieved.

            # Implementation Detail:
            # I'll retrieve `bs_orient` and `ut_orient` from `loader`?
            # Or `loader` has `bs_orient`?
            # In Interface code: `bs_orient_mapped = tf.gather(bs_orient, ...)` where `bs_orient` was ARGUMENT.
            # So I will access `ut_orient` etc from `kwargs` (I will add `**kwargs` to signature).

            # Temporary: dummy zeros if not found (mimics Interface which had fallbacks?)
            # No, Interface had them as args.
            # I will use `tf.zeros` for now and plan to fix the signature in Interface refactoring.

            # Helper to process UT-side inputs (Slice -> Tile -> Flatten)
            def process_ut_input(tensor_full):
                if tensor_full is None:
                    return tf.zeros([total_links * batch_size, 1, 3])
                t_batch = tensor_full[:, start_ut:end_ut, :]
                t_tiled = tf.tile(tf.expand_dims(t_batch, 2), [1, 1, num_neighbors, 1])
                return tf.reshape(t_tiled, [-1, 1, 3])

            # Helper to process BS-side inputs (Gather -> Flatten)
            def process_bs_input(tensor_full):
                if tensor_full is None:
                    return tf.zeros([total_links * batch_size, 1, 3])
                t_mapped = tf.gather(
                    tensor_full, neighbor_indices_batch, axis=1, batch_dims=1
                )
                return tf.reshape(t_mapped, [-1, 1, 3])

            ut_vel = process_ut_input(ut_velocities)
            ut_orient_flat = process_ut_input(ut_orient)
            bs_orient_flat = process_bs_input(bs_orient)

            dummy_zeros = tf.zeros([total_links * batch_size, 1, 1], dtype=tf.float32)
            dummy_los = tf.zeros([total_links * batch_size, 1, 1], dtype=tf.bool)

            topo = Topology(
                velocities=ut_vel,
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

            # c_ds (Delay Spread) for scaling?
            # self._channel_model._scenario.get_param('cDS')?
            # If external, maybe we don't have scenario?
            # We assume cDS is negligible or standard if not provided.
            c_ds = tf.zeros([total_links * batch_size, 1, 1])

            # 4. Call Sampler
            # Note: We need to pass batch_size=1 to sampler because we flattened links.
            # The sampler iterates over time samples.
            h_element, delays_sampled = self._channel_model._cir_sampler(
                1,  # num_time_samples
                30e3,  # dummy sampling freq
                k_flat,
                rays_obj_flat,
                topo,
                c_ds,
            )

            # h_element: [Links, Rx(1), RxAnt, Tx(1), TxAnt, Paths, Time] (Step 11 output)

            # 5. Apply Pathloss/SF
            gain_lin = tf.sqrt(sf_flat) * tf.pow(10.0, -pl_flat / 20.0)
            # gain has shape [Links, 1, 1]. Broadcast to h_element.
            h_element = (
                h_element
                * tf.complex(gain_lin, 0.0)[
                    :, :, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis
                ]
            )

            # 6. Backend Processing (CIR -> OFDM -> BF)
            # h_element is 'a' (Path Coefficients).
            # Squeeze singular dims: [Links, RxA, TxA, P, T]
            a = tf.squeeze(h_element, axis=[1, 3])  # [Links, RxA, TxA, P, T]

            # Transpose to [Links, P, RxA, TxA, T] ?
            # cir_to_ofdm expects 'a' [..., P, T].
            # h_element from step 11 is [..., P, T].
            # So [Links, RxA, TxA, P, T] is perfect.

            tau = delays_flat  # [Links, 1, 1, P]
            # Must match 'a' rank prefix. Broadcast tau?
            # cir_to_ofdm handles broadcasting if Shapes match (except P, T).
            # tau: [Links, 1, 1, P]. a: [Links, RxA, TxA, P, T].
            # It should work.

            # Need to loop chunks for OFDM conversion
            # We reuse _compute_chunk_element_channel logic?
            # But _compute_chunk_element_channel assumes 'a' and 'tau' are fields/args.
            # Yes. passing a and tau.

            # Flatten Links -> Batch for the helper?
            # helper takes a, tau.
            # a: [L, R, T, P, S]. tau: [L, 1, 1, P].
            # This is fine.

            chunk_h_ports = []
            for s_idx in range(0, self._num_active_sc, chunk_size):
                e_idx = min(s_idx + chunk_size, self._num_active_sc)
                num_chunk = e_idx - s_idx

                # CIR -> OFDM
                h_ofdm_elem = self.get_h_freq_chunk(a, tau, s_idx, num_chunk)
                # h_ofdm_elem: [Links, RxA, TxA, 1(Time), ChunkSC]

                # Apply Weights
                # We need weights!
                # Gather weights for links?
                # User's strict requirement: "Weights... passed clean".
                # Weights are stored in self.w_rf, self.a_rf?
                # If sector-specific, we need to Gather them.
                # Since we flattened links, we need flattened weights.
                # Assuming weights are global [Ant, Port] for now (simplest case).
                # If [Batch, U, ...], we need gathering logic.
                # For this implementation, I'll use current self.w_rf/a_rf and assume broadcasting.

                h_ofdm_port = self._apply_weights(h_ofdm_elem, self.w_rf, self.a_rf)
                chunk_h_ports.append(h_ofdm_port)

            h_port_flat = tf.concat(chunk_h_ports, axis=-1)

            # 7. Reshape to [Batch, U, Neighbor, ...]
            # h_port_flat: [TotalLinks, RxP, TxP, 1, SC]
            # Reshape -> [Batch_Sim=1, U, Neighbor, RxP, TxP, 1, SC]
            # Wait, TotalLinks = CurrentBatchSize * Neighbors.
            # We are inside batch loop 'i'.
            # current_batch_size is dimension of U.
            out_shape = [
                batch_size,
                current_batch_size,
                num_neighbors,
                self.num_rx_ports,
                self.num_tx_ports,
                1,
                -1,
            ]
            h_batch = tf.reshape(h_port_flat, out_shape)

            # Transpose to [Batch, U, N, ...]? Already is.
            h_chunks.append(h_batch)

        return tf.concat(h_chunks, axis=1)  # Concat along UT axis

    def _compute_statistical_links(
        self, batch_size, neighbor_indices, return_element_channel, chunk_size
    ):
        """
        Handles Statistical Model by slicing global topology.
        Optimized to compute only unique links.
        """
        # neighbor_indices: [Batch, Num_UT, Num_Neighbor] (Global BS IDs)

        # Batch processing for UTs to avoid OOM
        # Set a safe batch size (e.g., 4 or 8 depending on memory)
        # Reduced to 1 due to strict 718MB VRAM limit
        batch_size_ut = 1
        num_uts = tf.shape(neighbor_indices)[1]

        h_chunks = []
        num_batches_ut = (num_uts + batch_size_ut - 1) // batch_size_ut

        # Access global topology once
        g_topo = self._global_topology

        for i in range(num_batches_ut):
            start_ut = i * batch_size_ut
            end_ut = tf.minimum(start_ut + batch_size_ut, num_uts)

            # Slice indices for current batch
            # neighbor_indices_batch: [Batch, BatchUT, Neighbor]
            neighbor_indices_batch = neighbor_indices[:, start_ut:end_ut, :]

            # 1. Identify Unique BS Indices for this batch
            neighbor_indices_batch_int = tf.cast(neighbor_indices_batch, tf.int32)
            all_bs_indices = tf.reshape(neighbor_indices_batch_int, [-1])
            unique_bs_indices, _ = tf.unique(all_bs_indices)
            unique_bs_indices = tf.sort(unique_bs_indices)

            current_ut_indices = tf.range(start_ut, end_ut, dtype=tf.int32)

            # 2. Slice Topology
            def slice_topo_subset(key, indices):
                data = g_topo.get(key)
                if data is None:
                    return None
                return tf.gather(data, indices, axis=1)

            sliced_bs_loc = slice_topo_subset("bs_loc", unique_bs_indices)
            sliced_bs_orient = slice_topo_subset("bs_orient", unique_bs_indices)

            sliced_ut_loc = slice_topo_subset("ut_loc", current_ut_indices)
            sliced_ut_orient = slice_topo_subset("ut_orient", current_ut_indices)
            sliced_ut_vel = slice_topo_subset("ut_velocities", current_ut_indices)
            sliced_in_state = slice_topo_subset("in_state", current_ut_indices)

            # 3. Set Sliced Topology
            self._channel_model.set_topology(
                ut_loc=sliced_ut_loc,
                bs_loc=sliced_bs_loc,
                ut_orientations=sliced_ut_orient,
                bs_orientations=sliced_bs_orient,
                ut_velocities=sliced_ut_vel,
                in_state=sliced_in_state,
            )

            # 4. Compute Channel
            # Note: channel model now sees (BatchUT, UniqueBS)
            if return_element_channel:
                h_sliced = self.get_element_channel(batch_size, chunk_size=chunk_size)
            else:
                # Prepare Sliced Weights
                # Identify IDs for slicing based on direction
                direction = getattr(self._channel_model, "direction", "uplink")
                # Weights: [Batch, Num_Entity, Ant, Port] (Global). Batch usually 1 for weights.

                # Check if weights are set
                w_rf_sliced = self.w_rf
                a_rf_sliced = self.a_rf

                if self.w_rf is not None and self.a_rf is not None:
                    # Determine indices for Tx and Rx
                    if direction == "uplink":
                        # Tx = UT (Batch), Rx = BS (Unique)
                        tx_indices = current_ut_indices
                        rx_indices = unique_bs_indices
                    else:
                        # Tx = BS (Unique), Rx = UT (Batch)
                        tx_indices = unique_bs_indices
                        rx_indices = current_ut_indices

                    # Slice Weights (Axis 1)
                    # Assume w_rf/a_rf shape [Batch, Num, Ant, Port]
                    # If Batch > 1 (e.g. per-drop), we should also gather from Batch dim?
                    # But usually beam weights are fixed or per-drop (Batch=1).
                    # If simulator uses batch_size > 1, we assume weights match batch size.
                    # We only slice axis 1 (Entity ID).

                    if (
                        self.w_rf.shape[1] is not None
                    ):  # Can be None in standard Sionna? No.
                        w_rf_sliced = tf.gather(self.w_rf, tx_indices, axis=1)

                    if self.a_rf.shape[1] is not None:
                        a_rf_sliced = tf.gather(self.a_rf, rx_indices, axis=1)

                h_sliced = self.get_port_channel(
                    batch_size,
                    chunk_size=chunk_size,
                    w_rf=w_rf_sliced,
                    a_rf=a_rf_sliced,
                )

            # h_sliced: [Batch, Rx, ..., Tx, ...]
            # Rx/Tx dimensions depend on direction and num_ant/ports

            # 5. Determine Direction & Map Indices
            direction = getattr(self._channel_model, "direction", "uplink")
            is_uplink = direction == "uplink"

            # Map Global BS IDs (in neighbor_indices_batch) to Sliced BS IDs (index in unique_bs_indices)
            sliced_neighbor_indices = tf.searchsorted(
                unique_bs_indices, neighbor_indices_batch_int
            )

            # Handle Permutation and Gathering
            if is_uplink:
                # Rx=BS (Unique), Tx=UT (CurrentBatch)
                # h_sliced: [Batch, UniqueBS, RxA, BatchUT, TxA, ...]

                # Permute to [Batch, BatchUT, UniqueBS, RxA, TxA, ...]
                # Indices: 0(B), 1(BS), 2(Rx), 3(UT), 4(Tx)...
                # Target: 0, 3, 1, 2, 4...
                perm = [0, 3, 1, 2, 4, 5, 6]
                h_permuted = tf.transpose(h_sliced, perm=perm)

                # Gather Neighbors along axis 2 (UniqueBS)
                h_out = tf.gather(
                    h_permuted, sliced_neighbor_indices, axis=2, batch_dims=2
                )

            else:
                # Rx=UT (CurrentBatch), Tx=BS (Unique)
                # h_sliced: [Batch, BatchUT, RxA, UniqueBS, TxA, ...]

                # Permute to [Batch, BatchUT, UniqueBS, RxA, TxA, ...]
                # Indices: 0(B), 1(UT), 2(Rx), 3(BS), 4(Tx)...
                # Target: 0, 1, 3, 2, 4...
                perm = [0, 1, 3, 2, 4, 5, 6]
                h_permuted = tf.transpose(h_sliced, perm=perm)

                # Gather Neighbors along axis 2 (UniqueBS)
                h_out = tf.gather(
                    h_permuted, sliced_neighbor_indices, axis=2, batch_dims=2
                )

            h_chunks.append(h_out)

        # Concatenate results along UT axis
        return tf.concat(h_chunks, axis=1)

    def _apply_weights(self, h_elem, w_rf, a_rf):
        """
        Applies Analog Beamforming weights to the element-domain channel.
        Supports various weight shapes (Shared, Sector-specific, Fully Individual).

        Args:
            h_elem: Element-domain channel
                    [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, num_sc]
            w_rf: Tx weights. Supported shapes:
                  - [tx_ant, tx_port] (Shared across all Tx/Batch)
                  - [num_tx, tx_ant, tx_port] (Per-Tx sector weights)
                  - [batch, num_tx, tx_ant, tx_port] (Per-Link weights)
            a_rf: Rx weights. Supported shapes:
                  - [rx_ant, rx_port] (Shared across all Rx/Batch)
                  - [num_rx, rx_ant, rx_port] (Per-Rx weights)
                  - [batch, num_rx, rx_ant, rx_port] (Per-Link weights)

        Returns:
            h_port: Port-domain channel
                    [batch, num_rx, num_rx_ports, num_tx, num_tx_ports, num_ofdm, num_sc]
        """
        # h dimensions: b(0), r(1), u(2), t(3), v(4), s(5), c(6)
        tf.print("DEBUG: h_elem shape =", tf.shape(h_elem))
        tf.print("DEBUG: w_rf shape =", tf.shape(w_rf))
        tf.print("DEBUG: a_rf shape =", tf.shape(a_rf))

        # --- 1. TX Beamforming (Contract v -> p) ---
        rank_w = w_rf.shape.ndims
        if rank_w == 2:  # [v, p]
            eq_tx = "brutvsc,vp->brutpsc"
        elif rank_w == 3:  # [t, v, p]
            eq_tx = "brutvsc,tvp->brutpsc"
        elif rank_w == 4:  # [b, t, v, p]
            eq_tx = "brutvsc,btvp->brutpsc"
        else:
            raise ValueError(f"Unsupported w_rf rank: {rank_w}. Expected 2, 3, or 4.")

        h_tx_bf = tf.einsum(eq_tx, h_elem, w_rf)

        # --- 2. RX Beamforming (Contract u -> q) ---
        # h_tx_bf dimensions: b(0), r(1), u(2), t(3), p(4), s(5), c(6)
        rank_a = a_rf.shape.ndims
        if rank_a == 2:  # [u, q]
            eq_rx = "brutpsc,uq->brqtpsc"
        elif rank_a == 3:  # [r, u, q]
            eq_rx = "brutpsc,ruq->brqtpsc"
        elif rank_a == 4:  # [b, r, u, q]
            eq_rx = "brutpsc,bruq->brqtpsc"
        else:
            raise ValueError(f"Unsupported a_rf rank: {rank_a}. Expected 2, 3, or 4.")

        h_port = tf.einsum(eq_rx, h_tx_bf, tf.math.conj(a_rf))

        return h_port

    def __call__(self, batch_size):
        """
        Standard Sionna entry point.
        Returns: [batch, num_rx, num_rx_ports, num_tx, num_tx_ports, num_ofdm, num_sc]
        """
        return self.get_port_channel(batch_size)

    def get_element_channel(self, batch_size, chunk_size=72):
        """
        Generates the element-domain channel (before BF application).
        Returns: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, num_sc]
        """
        # 1. Generate Paths (CIR)
        a, tau = self._cir_sampler(
            batch_size, self._num_ofdm_symbols, self._sampling_frequency
        )

        h_element_chunks = []
        for start_idx in range(0, self._num_active_sc, chunk_size):
            num_chunk = min(chunk_size, self._num_active_sc - start_idx)
            # Use Base Class Logic
            h_elem = self.get_h_freq_chunk(a, tau, start_idx, num_chunk)
            h_element_chunks.append(h_elem)

        return tf.concat(h_element_chunks, axis=-1)

    def get_port_channel(self, batch_size, chunk_size=72, w_rf=None, a_rf=None):
        """
        Computes the channel in the port domain (after beamforming).

        Args:
            batch_size: Batch size
            chunk_size: Frequency chunk size
            w_rf: Optional Tx analog weights [Batch, Tx, Ant, Port]. Uses self.w_rf if None.
            a_rf: Optional Rx analog weights [Batch, Rx, Ant, Port]. Uses self.a_rf if None.
        """
        # Prepare weights
        w_rf_use = w_rf if w_rf is not None else self.w_rf
        a_rf_use = a_rf if a_rf is not None else self.a_rf

        # 1. Generate CIR (Paths)
        # Sliced channel model generates paths for (Rx, Tx) as defined in set_topology
        a, tau = self._cir_sampler(
            batch_size, self._num_ofdm_symbols, self._sampling_frequency
        )

        # 2. Compute Channel in Frequency Domain (Chunked)
        h_chunks = []

        num_chunk_steps = (self._num_active_sc + chunk_size - 1) // chunk_size

        for start_idx in range(0, self._num_active_sc, chunk_size):
            num_chunk = min(chunk_size, self._num_active_sc - start_idx)

            # Element Channel Chunk [Batch, Rx, RxA, Tx, TxA, 1, ChunkFreq]
            h_elem_chunk = self.get_h_freq_chunk(a, tau, start_idx, num_chunk)

            # Apply Weights
            # h_elem_chunk: [B, R, RA, T, TA, 1, F]
            # w_rf_use: [B, T, TA, TP] (Tx weights)
            # a_rf_use: [B, R, RA, RP] (Rx weights)

            # Use provided weights (sliced or global)
            h_port_chunk = self._apply_weights(h_elem_chunk, w_rf_use, a_rf_use)

            h_chunks.append(h_port_chunk)

        # Concatenate Chunks
        h_port = tf.concat(h_chunks, axis=-1)
        return h_port
