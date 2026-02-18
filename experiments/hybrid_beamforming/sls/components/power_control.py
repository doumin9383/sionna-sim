import tensorflow as tf
import numpy as np
from sionna.phy.utils import db_to_lin, dbm_to_watt

try:
    from sionna.sys import open_loop_uplink_power_control, downlink_fair_power_control
except ImportError:
    open_loop_uplink_power_control = None
    downlink_fair_power_control = None


class PowerControl:
    """
    Power Control implementation handling both Uplink and Downlink.
    Supports switchable logic between 'sionna' (native) and 'custom'.
    """

    def __init__(self, p0=-80.0, alpha=0.8, p_power_class=23.0, method="sionna"):
        """
        Args:
            p0 (float): Base power in dBm (Uplink)
            alpha (float): Path loss compensation factor (0.0 to 1.0)
            p_power_class (float): UE power class in dBm (default: 23 dBm for Class 3)
            method (str): "sionna" or "custom". Defaults to "sionna".
        """
        self.p0 = p0
        self.alpha = alpha
        self.p_power_class = p_power_class
        self.method = method

    def calculate_uplink_power(self, path_loss_db, num_rbs, mpr_db=0.0):
        """
        Calculates Uplink Transmission Power (PUSCH).

        Args:
            path_loss_db (tf.Tensor): Path loss in dB
            num_rbs (int or tf.Tensor): Number of allocated Resource Blocks
            mpr_db (float or tf.Tensor): Maximum Power Reduction in dB

        Returns:
            tf.Tensor: Transmission power in dBm
        """
        # Ensure proper types
        path_loss_db = tf.cast(path_loss_db, tf.float32)
        mpr_db = tf.cast(mpr_db, tf.float32)

        # Calculates P_cmax = P_power_class - MPR
        p_cmax = self.p_power_class - mpr_db

        if self.method == "sionna" and open_loop_uplink_power_control is not None:
            # Flatten inputs to handle arbitrary batch dimensions (Sionna function requires rank 0 or 1)
            original_shape = tf.shape(path_loss_db)
            path_loss_flat = tf.reshape(path_loss_db, [-1])
            mpr_flat = tf.reshape(mpr_db, [-1] if tf.rank(mpr_db) > 0 else [])

            # Recalculate P_cmax for flat inputs
            # If mpr_flat is empty (scalar input broadcasted), it uses scalar subtraction
            if tf.rank(mpr_db) > 0:
                p_cmax_flat = self.p_power_class - mpr_flat
            else:
                p_cmax_flat = self.p_power_class - mpr_db  # Scalar

            # Sionna expects linear pathloss
            path_loss_lin_flat = db_to_lin(path_loss_flat)

            # num_allocated_subcarriers = num_rbs * 12
            num_subcarriers = tf.cast(num_rbs, tf.float32) * 12.0

            # Broadcast scalar num_subcarriers to match pathloss shape
            if tf.rank(num_subcarriers) == 0:
                num_subcarriers = tf.broadcast_to(
                    num_subcarriers, tf.shape(path_loss_flat)
                )
            elif tf.rank(num_subcarriers) > 0:
                num_subcarriers = tf.reshape(num_subcarriers, [-1])

            # Call Sionna function
            # Output is in Watts, we need dBm
            p_tx_watt_flat = open_loop_uplink_power_control(
                pathloss=path_loss_lin_flat,
                num_allocated_subcarriers=num_subcarriers,
                alpha=self.alpha,
                p0_dbm=self.p0,
                ut_max_power_dbm=p_cmax_flat,  # We pass adjusted P_cmax as max power
            )

            # Avoid log(0)
            p_tx_dbm_flat = (
                10.0
                * tf.math.log(tf.maximum(p_tx_watt_flat, 1e-20))
                / tf.math.log(10.0)
                + 30.0
            )

            # Reshape back to original shape
            p_tx_dbm = tf.reshape(p_tx_dbm_flat, original_shape)

            return p_tx_dbm

        else:
            # Custom Logic (Fallback)
            # 10 * log10(M_RB)
            # Bandwidth factor is 10*log10(num_rbs) in original code?
            # Original code: 10.0 * np.log10(max(1, num_rbs))
            # Wait, original code treated num_rbs as the multiplier?
            # "10 * log10(M_RB)" usually refers to Allocation Bandwidth.
            # If num_rbs is number of RBs, then yes.

            bandwidth_factor = (
                10.0
                * tf.math.log(tf.maximum(tf.cast(num_rbs, tf.float32), 1.0))
                / tf.math.log(10.0)
            )
            p_open_loop = self.p0 + bandwidth_factor + self.alpha * path_loss_db

            p_tx = tf.minimum(p_cmax, p_open_loop)
            return p_tx

    def calculate_downlink_power(
        self, path_loss_db, bs_max_power_dbm=43.0, num_users=1, fairness=0.0
    ):
        """
        Calculates Downlink Transmission Power.
        Currently a placeholder or wrapper for fair_power_control.

        Args:
            path_loss_db (tf.Tensor): Path loss in dB
            bs_max_power_dbm (float): Total BS power
            num_users (int): Number of users to share power
            fairness (float): Fairness parameter (0=Sum Rate, 1=Proportional Fair)

        Returns:
            tf.Tensor: Tx Power per UT in dBm
        """
        # For now, simplistic equal power split or full power if single user
        # p_tx_dbm = bs_max_power_dbm - 10*log10(num_users)
        # return p_tx_dbm * ones_like(path_loss_db)

        # If we want to use 'downlink_fair_power_control', we need 'interference_plus_noise'.
        # Since we don't have it easily here, we skip it for now or implement equal split.

        # Simple Equal Split
        p_per_user_dbm = bs_max_power_dbm - 10.0 * np.log10(max(1, num_users))
        return tf.fill(tf.shape(path_loss_db), p_per_user_dbm)
