import sionna
import tensorflow as tf
# Additional external libraries
import matplotlib.pyplot as plt
import numpy as np

# Sionna components
from sionna.sys.utils import spread_across_subcarriers
from sionna.sys import (
    PHYAbstraction,
    OuterLoopLinkAdaptation,
    gen_hexgrid_topology,
    get_pathloss,
    open_loop_uplink_power_control,
    downlink_fair_power_control,
    get_num_hex_in_grid,
    PFSchedulerSUMIMO,
)
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.utils import db_to_lin, dbm_to_watt, log2, insert_dims
from sionna.phy import config, dtypes, Block
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, PanelArray
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid,
    RZFPrecodedChannel,
    EyePrecodedChannel,
    LMMSEPostEqualizationSINR,
)

# Set random seed for reproducibility
sionna.phy.config.seed = 42

# Internal computational precision
sionna.phy.config.precision = "single"  # 'single' or 'double'


def estimate_achievable_rate(sinr_eff_db_last, num_ofdm_sym, num_subcarriers):
    """Estimate achievable rate"""
    # [batch_size, num_bs, num_ut_per_sector]
    rate_achievable_est = log2(
        tf.cast(1, sinr_eff_db_last.dtype) + db_to_lin(sinr_eff_db_last)
    )

    # Broadcast to time/frequency grid
    # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector]
    rate_achievable_est = insert_dims(rate_achievable_est, 2, axis=-2)
    rate_achievable_est = tf.tile(
        rate_achievable_est, [1, 1, num_ofdm_sym, num_subcarriers, 1]
    )
    return rate_achievable_est
