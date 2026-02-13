import tensorflow as tf
import numpy as np
from sionna.phy.channel.tr38901 import PanelArray
from .channel_engines import ChannelPhysicsEngine, ChunkedGenerateOFDMChannel


class GenerateHybridBeamformingOFDMChannel(
    ChannelPhysicsEngine, ChunkedGenerateOFDMChannel
):
    """
    Sionna-compatible Block that generates a Digital Port Channel by applying
    Analog Beamforming to an underlying physical channel.
    Inherits Pure Physics Engine for core computations.
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
        # Initialize ChunkedGenerateOFDMChannel (which initializes Block and GenerateOFDMChannel)
        super().__init__(channel_model, resource_grid, normalize_channel)

        self.tx_array = tx_array
        self.rx_array = rx_array
        self.num_tx_ports = num_tx_ports
        self.num_rx_ports = num_rx_ports

        # Extract active frequencies for efficient chunking
        self._active_frequencies = tf.gather(
            self._all_frequencies, resource_grid.effective_subcarrier_ind
        )
        self._num_active_sc = len(self._active_frequencies)

        # Global Topology Storage for statistical models
        self._global_topology = None

        # Default weights
        self._init_default_weights()

    def _init_default_weights(self):
        self.w_rf = tf.eye(
            self.tx_array.num_ant, num_columns=self.num_tx_ports, dtype=tf.complex64
        )
        self.a_rf = tf.eye(
            self.rx_array.num_ant, num_columns=self.num_rx_ports, dtype=tf.complex64
        )

    def set_analog_weights(self, w_rf, a_rf):
        self.w_rf = w_rf
        self.a_rf = a_rf

    def set_topology(
        self, ut_loc, bs_loc, ut_orient, bs_orient, ut_velocities, in_state, store=True
    ):
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
        chunk_size=36,
        **topology_kwargs,
    ):
        """
        Main entry point for ID-based sparse calculation.
        """
        if external_loader is not None:
            return self._compute_external_links(
                batch_size,
                neighbor_indices,
                external_loader,
                return_element_channel,
                chunk_size,
                **topology_kwargs,
            )

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
        ID-based external data processing. Bypasses set_topology.
        """
        from sionna.phy.channel.tr38901 import Rays, Topology

        num_ut_total = tf.shape(neighbor_indices)[1]
        num_neighbors = tf.shape(neighbor_indices)[2]

        # UT-side batching to save memory
        batch_size_ut = 1
        h_chunks_ut = []

        for i in range((num_ut_total + batch_size_ut - 1) // batch_size_ut):
            start_ut = i * batch_size_ut
            end_ut = tf.minimum(start_ut + batch_size_ut, num_ut_total)
            ut_indices_slice = tf.range(start_ut, end_ut)
            neighbor_indices_slice = neighbor_indices[:, start_ut:end_ut, :]

            # Fetch data from external loader
            data = loader.get_rays(ut_indices=ut_indices_slice, bs_indices=None)

            def gather_neighbor_data(tensor):
                return tf.gather(tensor, neighbor_indices_slice, axis=2, batch_dims=2)

            # Unpack Rays
            rays_obj = Rays(
                delays=gather_neighbor_data(data["delays"]),
                powers=gather_neighbor_data(data["powers"]),
                aoa=gather_neighbor_data(data["aoa"]),
                aod=gather_neighbor_data(data["aod"]),
                zoa=gather_neighbor_data(data["zoa"]),
                zod=gather_neighbor_data(data["zod"]),
                xpr=gather_neighbor_data(data["xpr"]),
            )

            # Unpack LSPs
            pl = tf.gather(
                tf.transpose(data["pathloss"], [0, 2, 1]),
                neighbor_indices_slice,
                axis=2,
                batch_dims=2,
            )
            sf = tf.gather(
                tf.transpose(data["shadow_fading"], [0, 2, 1]),
                neighbor_indices_slice,
                axis=2,
                batch_dims=2,
            )
            kf = tf.gather(
                tf.transpose(data["k_factor"], [0, 2, 1]),
                neighbor_indices_slice,
                axis=2,
                batch_dims=2,
            )

            # Flatten to links for pure physics engine
            total_links = (end_ut - start_ut) * num_neighbors

            def flatten_links(t):
                s = tf.shape(t)
                return tf.reshape(t, tf.concat([[-1, 1, 1], s[3:]], axis=0))

            rays_flat = Rays(
                delays=flatten_links(rays_obj.delays),
                powers=flatten_links(rays_obj.powers),
                aoa=flatten_links(rays_obj.aoa),
                aod=flatten_links(rays_obj.aod),
                zoa=flatten_links(rays_obj.zoa),
                zod=flatten_links(rays_obj.zod),
                xpr=flatten_links(rays_obj.xpr),
            )

            # Construct Dummy Topology for steering vector calculation
            # Extract from topology_kwargs (passed from Interface)
            ut_orient = tf.tile(
                tf.expand_dims(topology_kwargs["ut_orient"][:, start_ut:end_ut, :], 2),
                [1, 1, num_neighbors, 1],
            )
            bs_orient = tf.gather(
                topology_kwargs["bs_orient"],
                neighbor_indices_slice,
                axis=1,
                batch_dims=1,
            )
            ut_vel = tf.tile(
                tf.expand_dims(
                    topology_kwargs["ut_velocities"][:, start_ut:end_ut, :], 2
                ),
                [1, 1, num_neighbors, 1],
            )

            topo_flat = Topology(
                velocities=tf.reshape(ut_vel, [-1, 1, 3]),
                tx_orientations=tf.reshape(bs_orient, [-1, 1, 3]),
                rx_orientations=tf.reshape(ut_orient, [-1, 1, 3]),
                moving_end="rx",
                los=tf.zeros([total_links, 1, 1], dtype=tf.bool),
                distance_3d=tf.zeros([total_links, 1, 1]),
                los_aoa=tf.zeros([total_links, 1, 1]),
                los_aod=tf.zeros([total_links, 1, 1]),
                los_zoa=tf.zeros([total_links, 1, 1]),
                los_zod=tf.zeros([total_links, 1, 1]),
            )

            # Call Engine Sampler
            # Note: _cir_sampler is internal to TR38901 models.
            # We assume self._channel_model is a TR38901 instance.
            h_element, _ = self._channel_model._cir_sampler(
                1,
                30e3,
                tf.reshape(kf, [-1]),
                rays_flat,
                topo_flat,
                tf.zeros([total_links, 1, 1]),
            )

            # Apply Pathloss
            gain = tf.sqrt(tf.reshape(sf, [-1])) * tf.pow(
                10.0, -tf.reshape(pl, [-1]) / 20.0
            )
            h_element *= tf.complex(gain, 0.0)[
                :,
                tf.newaxis,
                tf.newaxis,
                tf.newaxis,
                tf.newaxis,
                tf.newaxis,
                tf.newaxis,
            ]

            # CIR to OFDM (Chunked)
            a = tf.squeeze(h_element, axis=[1, 3])  # [Links, RxA, TxA, P, T]
            tau = rays_flat.delays  # [Links, 1, 1, P]

            h_ports_chunks = []
            for s_idx in range(0, self._num_active_sc, chunk_size):
                num_c = min(chunk_size, self._num_active_sc - s_idx)
                # Call Engine Core
                h_elem_chunk = self.get_h_freq_chunk(
                    a, tau, s_idx, num_c, self._active_frequencies
                )

                if return_element_channel:
                    h_ports_chunks.append(h_elem_chunk)
                else:
                    # Apply Weights (Port Domain)
                    direction = getattr(self._channel_model, "direction", "uplink")
                    if direction == "uplink":
                        w_use = tf.gather(
                            tf.tile(
                                tf.expand_dims(self.w_rf, 2),
                                [1, 1, num_neighbors, 1, 1],
                            ),
                            ut_indices_slice,
                            axis=1,
                        )  # [B, U, N, TA, TP]
                        a_use = tf.gather(
                            self.a_rf, neighbor_indices_slice, axis=1, batch_dims=1
                        )  # [B, U, N, RA, RP]
                    else:
                        w_use = tf.gather(
                            self.w_rf, neighbor_indices_slice, axis=1, batch_dims=1
                        )
                        a_use = tf.gather(
                            tf.tile(
                                tf.expand_dims(self.a_rf, 2),
                                [1, 1, num_neighbors, 1, 1],
                            ),
                            ut_indices_slice,
                            axis=1,
                        )

                    # Reshape weights to [Links, Ant, Port] (Rank 3 for Link-specific application)
                    w_use_flat = tf.reshape(
                        w_use, [-1, self.tx_array.num_ant, self.num_tx_ports]
                    )
                    a_use_flat = tf.reshape(
                        a_use, [-1, self.rx_array.num_ant, self.num_rx_ports]
                    )

                    h_port_chunk = self._apply_weights(
                        h_elem_chunk, w_use_flat, a_use_flat
                    )
                    h_ports_chunks.append(h_port_chunk)

            h_batch_ut = tf.concat(h_ports_chunks, axis=-1)
            # Reshape back to [Batch, UT_slice, Neighbors, ...]
            new_shape = [
                batch_size,
                end_ut - start_ut,
                num_neighbors,
            ] + h_batch_ut.shape[1:].as_list()
            h_chunks_ut.append(tf.reshape(h_batch_ut, new_shape))

        return tf.concat(h_chunks_ut, axis=1)

    def _compute_statistical_links(
        self, batch_size, neighbor_indices, return_element_channel, chunk_size
    ):
        """
        Statistical model path with topology slicing.
        """
        num_uts = tf.shape(neighbor_indices)[1]
        h_chunks = []

        for i in range((num_uts + 1 - 1) // 1):
            start_ut = i
            end_ut = i + 1
            idx_slice = neighbor_indices[:, start_ut:end_ut, :]

            # Identify unique BS for this UT (Neighbor 0 is always unique, but neighbors might overlap)
            unique_bs, _ = tf.unique(tf.reshape(tf.cast(idx_slice, tf.int32), [-1]))
            unique_bs = tf.sort(unique_bs)

            # Slice and Set Topology
            self.set_topology(
                ut_loc=tf.gather(
                    self._global_topology["ut_loc"], tf.range(start_ut, end_ut), axis=1
                ),
                bs_loc=tf.gather(self._global_topology["bs_loc"], unique_bs, axis=1),
                ut_orient=tf.gather(
                    self._global_topology["ut_orient"],
                    tf.range(start_ut, end_ut),
                    axis=1,
                ),
                bs_orient=tf.gather(
                    self._global_topology["bs_orient"], unique_bs, axis=1
                ),
                ut_velocities=tf.gather(
                    self._global_topology["ut_velocities"],
                    tf.range(start_ut, end_ut),
                    axis=1,
                ),
                in_state=tf.gather(
                    self._global_topology["in_state"],
                    tf.range(start_ut, end_ut),
                    axis=1,
                ),
                store=False,
            )

            # Generate and Map
            if return_element_channel:
                h_raw = self.get_element_channel(batch_size, chunk_size=chunk_size)
            else:
                # Slice weights for this local topology
                is_uplink = (
                    getattr(self._channel_model, "direction", "uplink") == "uplink"
                )

                # Helper to slice weights if they have Entity dimension (Rank 4)
                def slice_weight(w, indices, is_tx=True):
                    # BS-side weights: [B, nBS, Ant, Port]
                    # UT-side weights: [B, nUT, Ant, Port]
                    # Default weights: [Ant, Port]
                    if len(w.shape) == 4:
                        return tf.gather(w, indices, axis=1)
                    return w

                if is_uplink:
                    # Tx: UT (start_ut:end_ut), Rx: BS (unique_bs)
                    w_local = slice_weight(
                        self.w_rf, tf.range(start_ut, end_ut), is_tx=True
                    )
                    a_local = slice_weight(self.a_rf, unique_bs, is_tx=False)
                else:
                    # Tx: BS (unique_bs), Rx: UT (start_ut:end_ut)
                    w_local = slice_weight(self.w_rf, unique_bs, is_tx=True)
                    a_local = slice_weight(
                        self.a_rf, tf.range(start_ut, end_ut), is_tx=False
                    )

                h_raw = self._get_port_channel_with_weights(
                    batch_size, w_local, a_local, chunk_size=chunk_size
                )

            # Map back to [Batch, 1, Neighbors, ...]
            is_uplink = getattr(self._channel_model, "direction", "uplink") == "uplink"
            mapped_idx = tf.searchsorted(unique_bs, tf.cast(idx_slice, tf.int32))

            if is_uplink:
                # h_raw: [B, nRx=UniqueBS, RP, nTx=1, TP, Time, Freq]
                # mapped_idx: [B, 1, Neighbors]
                # h_out: [B, 1, Neighbors, RP, 1, TP, Time, Freq] (Rank 8)
                h_out = tf.gather(h_raw, mapped_idx, axis=1, batch_dims=1)
                h_out = tf.squeeze(
                    h_out, axis=4
                )  # Remove nTx=1 -> [B, 1, N, RP, TP, T, F]
            else:
                # h_raw: [B, nRx=1, RP, nTx=UniqueBS, TP, Time, Freq]
                # h_out after gather: [B, 1, RP, 1, Neighbors, TP, Time, Freq]
                h_out = tf.gather(h_raw, mapped_idx, axis=3, batch_dims=1)
                h_out = tf.squeeze(h_out, axis=1)  # Remove nRx=1
                # Final shape should be [B, 1, Neighbors, RP, TP, T, F]

            h_chunks.append(h_out)

        return tf.concat(h_chunks, axis=1)

    def _apply_weights(self, h_elem, w_rf, a_rf):
        # h_elem: [..., RA, TA, Time, Freq]
        rank = len(h_elem.shape)
        a_conj = tf.math.conj(a_rf)

        if rank == 7:
            # Sionna Standard 7D: [Batch, nRx, RxAnt, nTx, TxAnt, Time, Freq]
            # Normalize Weights to [B, nEnt, Ant, Port, ...]
            # Defaults are [Ant, Port], Simulator gives [B, nEnt, Ant, Port]
            w_tx = w_rf
            while len(w_tx.shape) < 4:
                w_tx = tf.expand_dims(w_tx, 0)
            w_rx = a_conj
            while len(w_rx.shape) < 4:
                w_rx = tf.expand_dims(w_rx, 0)

            # Equation: brQtPmf = conj(brRQ...) * brRtTmf * btTP...
            h_port = tf.einsum("brRtTmf,btTP...,brRQ...->brQtPmf", h_elem, w_tx, w_rx)

        elif rank == 5:
            # Flattened Links 5D: [Links, RxAnt, TxAnt, Time, Freq]
            w_tx = w_rf
            while len(w_tx.shape) < 3:
                w_tx = tf.expand_dims(w_tx, 0)
            w_rx = a_conj
            while len(w_rx.shape) < 3:
                w_rx = tf.expand_dims(w_rx, 0)

            # Equation: bQPTmf = conj(bRQ...) * bRTmf * bTP...
            h_port = tf.einsum("lRTmf,lTP...,lRQ...->lQPTmf", h_elem, w_tx, w_rx)
        else:
            raise ValueError(f"Unsupported channel rank for weight application: {rank}")

        return h_port

    def get_element_channel(self, batch_size, chunk_size=36):
        a, tau = self.get_paths(batch_size)
        chunks = []
        for s in range(0, self._num_active_sc, chunk_size):
            chunks.append(
                self.get_h_freq_chunk(
                    a,
                    tau,
                    s,
                    min(chunk_size, self._num_active_sc - s),
                    self._active_frequencies,
                )
            )
        return tf.concat(chunks, axis=-1)

    def get_port_channel(self, batch_size, chunk_size=36):
        return self._get_port_channel_with_weights(
            batch_size, self.w_rf, self.a_rf, chunk_size=chunk_size
        )

    def _get_port_channel_with_weights(self, batch_size, w_rf, a_rf, chunk_size=36):
        a, tau = self.get_paths(batch_size)
        chunks = []
        for s in range(0, self._num_active_sc, chunk_size):
            h_elem = self.get_h_freq_chunk(
                a,
                tau,
                s,
                min(chunk_size, self._num_active_sc - s),
                self._active_frequencies,
            )
            chunks.append(self._apply_weights(h_elem, w_rf, a_rf))
        return tf.concat(chunks, axis=-1)

    def __call__(self, batch_size):
        return self.get_port_channel(batch_size)
