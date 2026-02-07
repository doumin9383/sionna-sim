import tensorflow as tf
import numpy as np
from sionna.phy.channel import (
    GenerateOFDMChannel,
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)
from sionna.phy.channel.tr38901 import PanelArray


class ChunkedOFDMChannel(GenerateOFDMChannel):
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


class HybridOFDMChannel(GenerateOFDMChannel):
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

    def _apply_weights(self, h_elem, w_rf, a_rf):
        """
        Applies weights with broadcasting.
        h_elem: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, num_sc]
        w_rf: [..., tx_ant, tx_port]
        a_rf: [..., rx_ant, rx_port]
        """
        # 1. TX Beamforming (v -> p)
        # h: b r u t v s c
        # w: possible shapes: [v, p], [t, v, p], [b, t, v, p]

        # Determine equation based on w rank
        if len(w_rf.shape) == 2:  # [v, p]
            eq_tx = "brutvsc,vp->brutpsc"
        elif len(w_rf.shape) == 3:  # [t, v, p]
            eq_tx = "brutvsc,tvp->brutpsc"
        else:  # [b, t, v, p] - assumes full match or correct setup
            eq_tx = "brutvsc,btvp->brutpsc"

        h_tx_bf = tf.einsum(eq_tx, h_elem, w_rf)

        # 2. RX Beamforming (u -> q)
        # h_tx_bf: b r u t p s c
        # a: possible shapes: [u, q], [r, u, q], [b, r, u, q]

        if len(a_rf.shape) == 2:  # [u, q]
            eq_rx = "brutpsc,uq->brqtpsc"
        elif len(a_rf.shape) == 3:  # [r, u, q]
            eq_rx = "brutpsc,ruq->brqtpsc"
        else:  # [b, r, u, q]
            eq_rx = "brutpsc,bruq->brqtpsc"

        h_port = tf.einsum(eq_rx, h_tx_bf, tf.math.conj(a_rf))

        return h_port

    def __call__(self, batch_size):
        """
        Standard Sionna entry point.
        Returns: [batch, num_rx, num_rx_ports, num_tx, num_tx_ports, num_ofdm, num_sc]
        """
        return self.get_port_channel(batch_size)

    def get_port_channel(self, batch_size, chunk_size=72):
        """
        Internal implementation of chunked channel generation + BF application.
        """
        # 1. Generate Paths (CIR)
        # Inherited from GenerateOFDMChannel's setup
        a, tau = self._cir_sampler(
            batch_size, self._num_ofdm_symbols, self._sampling_frequency
        )

        # 2. Chunk-based processing loop over ACTIVE subcarriers
        h_port_chunks = []

        for start_idx in range(0, self._num_active_sc, chunk_size):
            end_idx = min(start_idx + chunk_size, self._num_active_sc)

            # A. Convert CIR to Frequency domain for this chunk of active SCs
            chunk_freqs = self._active_frequencies[start_idx:end_idx]
            # Ensure output is [..., num_time_steps, num_frequencies]
            h_elem = cir_to_ofdm_channel(
                chunk_freqs, a, tau, normalize=self._normalize_channel
            )

            # B. Apply Analog Beamforming (Einsum with broadcasting)
            h_port = self._apply_weights(h_elem, self.w_rf, self.a_rf)

            h_port_chunks.append(h_port)

        # 3. Concatenate and return
        return tf.concat(h_port_chunks, axis=-1)
