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


class HybridOFDMChannel:
    """
    Applies Analog Beamforming to generate Digital Port Channel.
    """

    def __init__(self, channel_model, resource_grid, tx_array, rx_array):
        self._chunked_channel = ChunkedOFDMChannel(channel_model, resource_grid)
        self.resource_grid = resource_grid
        self.tx_array = tx_array
        self.rx_array = rx_array

        # Initialize Weights (Identity/Diagonal for now - assuming fully digital or specific mapping)
        # For true hybrid, we need the mapping.
        # If tx_array is PanelArray, total elements = rows*cols*pol*panels
        # We assume a simple structure for MVP:
        #   num_ports < num_elements
        #   weights: [num_elements, num_ports]
        self._init_default_weights()

    def _init_default_weights(self):
        # Create dummy Identity-like weights if ports == elements, else random complex for test
        # Need to know dimensions.
        # Using heuristic:
        # tx_array.num_ant is total elements.
        pass

    def set_analog_weights(self, w_rf, a_rf):
        # w_rf: [batch, num_tx, num_tx_ant, num_tx_ports] or [num_tx_ant, num_tx_ports]
        # a_rf: [batch, num_rx, num_rx_ant, num_rx_ports]
        self.w_rf = w_rf
        self.a_rf = a_rf

    def get_port_channel(self, batch_size, num_tx_ports, num_rx_ports, chunk_size=72):
        """
        Generates the effective channel H_port: [batch, num_rx, num_rx_ports, num_tx, num_tx_ports, num_ofdm, num_sc]
        """
        # 1. Generate Paths (Global for this snapshot)
        a, tau = self._chunked_channel.get_paths(batch_size)

        # 2. Prepare storage for chunks
        h_port_chunks = []
        total_sc = (
            self.resource_grid.fft_size
        )  # Note: ResourceGrid might have guard bands?
        # Usually Sionna ResourceGrid only models occupied subcarriers or full FFT?
        # Check subcarrier indices. For simple setups, it's often full FFT or data SCs.
        # We rely on subcarrier_frequencies output size.

        # Safely determine loop range
        # Note: subcarrier_frequencies return size = fft_size usually.
        # But cir_to_ofdm_channel output freq dim matches input frequencies size.

        num_sc_total = self.resource_grid.fft_size  # Assuming full grid for channel gen

        # Define Analog Weights if not set (Mocking for now)
        # W_RF: [num_tx_ant, num_tx_ports]
        # A_RF: [num_rx_ant, num_rx_ports]
        # We assume batch dimension might exist or broadcast.
        if not hasattr(self, "w_rf"):
            # Create simple per-antenna mapping (Diagonal)
            # Assume num_tx_ports divides num_tx_ant? Or just 1 port per pol?
            # For MVP, let's make W_RF Identity if counts match, else Block Diagonal
            self.w_rf = tf.eye(
                self.tx_array.num_ant, num_columns=num_tx_ports, dtype=tf.complex64
            )
            self.a_rf = tf.eye(
                self.rx_array.num_ant, num_columns=num_rx_ports, dtype=tf.complex64
            )

        for start_idx in range(0, num_sc_total, chunk_size):
            end_idx = min(start_idx + chunk_size, num_sc_total)
            current_chunk_size = end_idx - start_idx

            # A. Get Element Channel Chunk
            # Shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps, num_chunk]
            # Wait, cir_to_ofdm_channel output: [..., num_ofdm_symbols, num_subcarriers]
            h_elem = self._chunked_channel.get_h_freq_chunk(
                a, tau, start_idx, current_chunk_size
            )

            # Shape check: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, num_chunk]

            # B. Apply Analog Beamforming
            # H_port = A_RF^H @ H_elem @ W_RF
            # A_RF shape: [num_rx_ant, num_rx_ports] (assume shared across batch/rx for now or broadcast)
            # W_RF shape: [num_tx_ant, num_tx_ports]

            # We use einsum.
            # Dimensions of h_elem: b(atch), r(x), u(rx_ant), t(x), v(tx_ant), s(ym), c(hunk)
            # Dimensions of w_rf: v, p(tx_port)
            # Dimensions of a_rf: u, q(rx_port)

            # Contraction:
            # sum_u sum_v (conj(a_rf[u,q]) * h_elem[b,r,u,t,v,s,c] * w_rf[v,p])
            # -> [b, r, q, t, p, s, c]

            # Note: w_rf, a_rf might be batched [b, r/t, ...] depending on implementation.
            # Assuming static for now (common to all users/BSs in this simpler view, OR passed in correctly).
            # If W_RF is BS-side, it might be [batch, num_tx_BS, num_ant, num_ports].

            # Let's assume broadacasting works.
            # Using specific indices for clarity.
            # h: bru tv sc
            # w: tvp (if batched/per-tx) or vp
            # a: ruq (if batched/per-rx) or uq

            # Applying TX BF first:
            # h @ w -> brutpsc
            h_tx_bf = tf.einsum("brutvsc,vp->brutpsc", h_elem, self.w_rf)

            # Applying RX BF:
            # a^H @ h
            # conj(a_rf) is needed.
            h_port = tf.einsum("uq,brutpsc->brqtpsc", tf.math.conj(self.a_rf), h_tx_bf)

            h_port_chunks.append(h_port)

        # 3. Concatenate
        h_port_full = tf.concat(h_port_chunks, axis=-1)
        return h_port_full
