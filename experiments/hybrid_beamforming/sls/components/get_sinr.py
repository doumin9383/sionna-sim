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


def get_sinr(
    tx_power,
    stream_management,
    no,
    direction,
    h_freq_fading,
    num_bs,
    num_ut_per_sector,
    num_streams_per_ut,
    resource_grid,
):
    """Compute post-equalization SINR. It is assumed:
     - DL: Regularized zero-forcing precoding
     - UL: No precoding, only power allocation
    LMMSE equalizer is used in both DL and UL.
    """
    # tx_power: [batch_size, num_bs, num_tx_per_sector,
    #            num_streams_per_tx, num_ofdm_sym, num_subcarriers]
    # Flatten across sectors
    # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers]
    s = tx_power.shape
    tx_power = tf.reshape(tx_power, [s[0], s[1] * s[2]] + s[3:])

    # Compute SINR
    # [batch_size, num_ofdm_sym, num_subcarriers, num_ut,
    #  num_streams_per_ut]
    if direction == "downlink":
        # Regularized zero-forcing precoding in the DL
        precoded_channel = RZFPrecodedChannel(
            resource_grid=resource_grid, stream_management=stream_management
        )
        h_eff = precoded_channel(
            h_freq_fading, tx_power=tx_power, alpha=no
        )  # Regularizer
    else:
        # No precoding in the UL: just power allocation
        precoded_channel = EyePrecodedChannel(
            resource_grid=resource_grid, stream_management=stream_management
        )
        h_eff = precoded_channel(h_freq_fading, tx_power=tx_power)

    # LMMSE equalizer
    lmmse_posteq_sinr = LMMSEPostEqualizationSINR(
        resource_grid=resource_grid, stream_management=stream_management
    )
    # Post-equalization SINR
    # [batch_size, num_ofdm_symbols, num_subcarriers, num_rx, num_streams_per_rx]
    sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)

    # [batch_size, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
    sinr = tf.reshape(
        sinr, sinr.shape[:-2] + [num_bs * num_ut_per_sector, num_streams_per_ut]
    )

    # Regroup by sector
    # [batch_size, num_ofdm_symbols, num_subcarriers, num_bs, num_ut_per_sector, num_streams_per_ut]
    sinr = tf.reshape(
        sinr, sinr.shape[:-2] + [num_bs, num_ut_per_sector, num_streams_per_ut]
    )

    # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector, num_streams_per_ut]
    sinr = tf.transpose(sinr, [0, 3, 1, 2, 4, 5])
    return sinr
