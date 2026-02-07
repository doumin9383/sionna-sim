import tensorflow as tf
import numpy as np


class PowerControl:
    """
    Uplink Power Control implementation (Open Loop + MPR constraint).

    Logic:
    P_tx = min(P_cmax, P_open_loop)
    P_cmax = P_power_class - MPR
    P_open_loop = P0 + 10 * log10(M_RB) + alpha * PL
    """

    def __init__(self, p0=-80.0, alpha=0.8, p_power_class=23.0):
        """
        Args:
            p0 (float): Base power in dBm
            alpha (float): Path loss compensation factor (0.0 to 1.0)
            p_power_class (float): UE power class in dBm (default: 23 dBm for Class 3)
        """
        self.p0 = p0
        self.alpha = alpha
        self.p_power_class = p_power_class

    def calculate_tx_power(self, path_loss_db, num_rbs, mpr_db=0.0):
        """
        Calculates the transmission power.

        Args:
            path_loss_db (float or tf.Tensor): Path loss in dB
            num_rbs (int): Number of allocated Resource Blocks
            mpr_db (float or tf.Tensor): Maximum Power Reduction in dB

        Returns:
            tf.Tensor: Transmission power in dBm
        """
        # Ensure inputs are tensors for batch processing if needed,
        # but robust enough for float inputs too.
        path_loss_db = tf.cast(path_loss_db, tf.float32)
        mpr_db = tf.cast(mpr_db, tf.float32)

        # 1. Calculate P_cmax
        p_cmax = self.p_power_class - mpr_db

        # 2. Calculate P_open_loop
        # 10 * log10(M_RB)
        # 1 RB = 12 subcarriers.
        # Assuming num_rbs is the number of RBs.
        # Bandwidth factor is 10*log10(num_rbs).
        # Note: If num_rbs is 0, this would verify -inf, but num_rbs should be >= 1.
        bandwidth_factor = 10.0 * np.log10(max(1, num_rbs))

        p_open_loop = self.p0 + bandwidth_factor + self.alpha * path_loss_db

        # 3. Calculate P_tx
        p_tx = tf.minimum(p_cmax, p_open_loop)

        return p_tx
