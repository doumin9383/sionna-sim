import tensorflow as tf
from sionna.phy import Block
from sionna.phy.utils import log2


class WaterFillingLinkAdaptation(Block):
    """
    Simplified Link Adaptation Block.
    Performs Water Filling Power Allocation on effective channels (singular values)
    and calculates Shannon throughput.
    """

    def __init__(self, resource_grid, transmitter, num_streams_per_tx, precision=None):
        super().__init__(precision=precision)
        self.resource_grid = resource_grid
        self.num_streams = num_streams_per_tx
        # We assume 'transmitter' object has power constraints, but we might just take max_power as arg for simplicity MVP.

    def water_filling(self, singular_values, noise_power, total_power):
        """
        Water Filling algorithm.

        Args:
            singular_values: [..., num_streams] Effective channel gains (lambda).
            noise_power: [..., num_streams] Noise + Interference power per stream.
            total_power: Total power constraint per transmission entity.

        Returns:
            p_alloc: [..., num_streams] Power allocated to each stream.
        """
        # Channel gains to power gains (lambda^2)
        channel_gains = tf.square(singular_values)

        # Avoid division by zero
        channel_gains = tf.maximum(channel_gains, 1e-12)

        # Noise-to-Channel Ratio (Inverse SNR level) = (N0 + I) / |h|^2
        inv_snr = noise_power / channel_gains

        # --- Batched Waterfilling with unique noise levels per stream ---
        shape = tf.shape(channel_gains)
        num_streams = shape[-1]

        # 1. Sort Inverse SNR (low to high)
        inv_snr_sorted = tf.sort(inv_snr, axis=-1)

        # 2. Find Water Level
        cumsum_inv_snr = tf.cumsum(inv_snr_sorted, axis=-1)
        k = tf.range(1, num_streams + 1, dtype=self.rdtype)
        mu_candidate = (total_power + cumsum_inv_snr) / k

        # Condition: Water level must be higher than the floor (inv_snr)
        active_mask = mu_candidate > inv_snr_sorted
        num_active = tf.reduce_sum(
            tf.cast(active_mask, tf.int32), axis=-1, keepdims=True
        )
        num_active = tf.maximum(num_active, 1)

        mu = tf.gather(
            mu_candidate, num_active - 1, batch_dims=len(mu_candidate.shape) - 1
        )

        # 3. Calculate power (ensuring mu is broadcastable)
        p_alloc = tf.maximum(mu - inv_snr, 0)

        return p_alloc

    def call(self, singular_values, noise_power, total_power):
        """
        Args:
            singular_values: Effective channel gains [..., num_streams]
            noise_power: N0 + Interference [..., num_streams]
            total_power: P_tx
        """
        # 1. Power Allocation
        p_alloc = self.water_filling(singular_values, noise_power, total_power)

        # 2. SINR = P * |h|^2 / (N0 + I)
        sinr = (p_alloc * tf.square(singular_values)) / tf.maximum(noise_power, 1e-15)

        return p_alloc, sinr

    def calculate_throughput(self, sinr):
        """
        Shannon Capacity: B * Sum(log2(1 + SINR))
        """
        # Sum over streams (last dimension)
        capacity_per_re = tf.reduce_sum(log2(1 + sinr), axis=-1)

        return capacity_per_re
