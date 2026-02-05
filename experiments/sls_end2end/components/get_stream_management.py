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


def get_stream_management(
    direction, num_rx, num_tx, num_streams_per_ut, num_ut_per_sector
):
    """
    Instantiate a StreamManagement object.
    It determines which data streams are intended for each receiver
    """
    if direction == "downlink":
        num_streams_per_tx = num_streams_per_ut * num_ut_per_sector
        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array(
            [
                [i1, i2]
                for i2 in range(num_tx)
                for i1 in np.arange(
                    i2 * num_ut_per_sector, (i2 + 1) * num_ut_per_sector
                )
            ]
        )
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    else:
        num_streams_per_tx = num_streams_per_ut
        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array(
            [
                [i1, i2]
                for i1 in range(num_rx)
                for i2 in np.arange(
                    i1 * num_ut_per_sector, (i1 + 1) * num_ut_per_sector
                )
            ]
        )
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    stream_management = StreamManagement(rx_tx_association, num_streams_per_tx)
    return stream_management
