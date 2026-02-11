import tensorflow as tf
from sionna.phy.channel import (
    GenerateOFDMChannel,
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)


class ChannelPhysicsEngine:
    """
    Mix-in for Sionna channel computation.
    Encapsulates internal functions like cir_to_ofdm_channel.
    """

    def _physics_engine_links(self, a, tau, frequencies, normalize_channel=None):
        """
        Core physics: CIR to OFDM conversion for a set of links.
        """
        # Fallback to self._normalize_channel if not provided
        normalize = (
            normalize_channel
            if normalize_channel is not None
            else getattr(self, "_normalize_channel", True)
        )
        return cir_to_ofdm_channel(frequencies, a, tau, normalize=normalize)

    def get_h_freq_chunk(self, a, tau, start_idx, num_chunk, frequencies):
        """
        Chunk-based frequency domain computation.
        """
        chunk_freqs = frequencies[start_idx : start_idx + num_chunk]
        return self._physics_engine_links(a, tau, chunk_freqs)


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
        """Generate paths (a, tau) for the current realization."""
        # Use IDENTIFIED attributes from Sionna GenerateOFDMChannel
        a, tau = self._channel_model(self._num_ofdm_symbols, self._sampling_frequency)
        return a, tau

    def get_h_freq_chunk(self, a, tau, start_idx, num_chunk):
        """Convert specific subcarriers from CIR to Frequency domain."""
        chunk_freqs = self._all_frequencies[start_idx : start_idx + num_chunk]
        h_freq_chunk = cir_to_ofdm_channel(
            chunk_freqs, a, tau, normalize=self._normalize_channel
        )
        return h_freq_chunk

    def get_rbg_channel(self, batch_size, rbg_size, active_rbgs=None):
        """Get channel sampled at RBG granularity."""
        a, tau = self.get_paths(batch_size)
        num_subcarriers = self._resource_grid.fft_size
        num_rbgs = tf.maximum(num_subcarriers // rbg_size, 1)

        if num_subcarriers < rbg_size:
            rbg_indices = tf.constant([num_subcarriers // 2], dtype=tf.int32)
        else:
            rbg_indices = tf.range(num_rbgs) * rbg_size + (rbg_size // 2)

        rbg_freqs = tf.gather(self._all_frequencies, rbg_indices)
        return cir_to_ofdm_channel(rbg_freqs, a, tau, normalize=self._normalize_channel)
