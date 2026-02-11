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


class GenerateHybridBeamformingOFDMChannel(GenerateOFDMChannel):
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
        # Initialize the base GenerateOFDMChannel
        super().__init__(channel_model, resource_grid, normalize_channel)

        self.tx_array = tx_array
        self.rx_array = rx_array
        self.num_tx_ports = num_tx_ports
        self.num_rx_ports = num_rx_ports

        # Prepare Active Frequencies based on ResourceGrid
        # Standard Sionna ResourceGrid might have guard bands or DC removal.
        all_freqs = subcarrier_frequencies(
            resource_grid.fft_size, resource_grid.subcarrier_spacing
        )
        self._active_frequencies = tf.gather(
            all_freqs, resource_grid.effective_subcarrier_ind
        )
        self._num_active_sc = len(self._active_frequencies)

        # Initialize Default Weights (Identity-like/Diagonal)
        self._init_default_weights()

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

    def set_topology(self, *args, **kwargs):
        """
        Proxy method to set topology on the underlying channel model.
        """
        self._channel_model.set_topology(*args, **kwargs)

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

    def _compute_chunk_element_channel(self, a, tau, start_idx, end_idx):
        """
        Helper to compute element domain channel for a specific frequency chunk.
        """
        chunk_freqs = self._active_frequencies[start_idx:end_idx]
        h_elem = cir_to_ofdm_channel(
            chunk_freqs, a, tau, normalize=self._normalize_channel
        )
        return h_elem

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
            end_idx = min(start_idx + chunk_size, self._num_active_sc)
            h_elem = self._compute_chunk_element_channel(a, tau, start_idx, end_idx)
            h_element_chunks.append(h_elem)

        return tf.concat(h_element_chunks, axis=-1)

    def get_port_channel(self, batch_size, chunk_size=72):
        """
        Generates the port-domain channel (after BF application) using chunking.
        """
        # 1. Generate Paths (CIR)
        a, tau = self._cir_sampler(
            batch_size, self._num_ofdm_symbols, self._sampling_frequency
        )

        # 2. Chunk-based processing loop over ACTIVE subcarriers
        h_port_chunks = []

        for start_idx in range(0, self._num_active_sc, chunk_size):
            end_idx = min(start_idx + chunk_size, self._num_active_sc)

            # A. Convert CIR to Frequency domain
            h_elem = self._compute_chunk_element_channel(a, tau, start_idx, end_idx)

            # B. Apply Analog Beamforming
            h_port = self._apply_weights(h_elem, self.w_rf, self.a_rf)

            h_port_chunks.append(h_port)

        # 3. Concatenate and return
        return tf.concat(h_port_chunks, axis=-1)
