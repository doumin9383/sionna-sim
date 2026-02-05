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


class ChannelMatrix(Block):
    def __init__(
        self, resource_grid, batch_size, num_rx, num_tx, coherence_time, precision=None
    ):
        super().__init__(precision=precision)
        self.resource_grid = resource_grid
        self.coherence_time = coherence_time
        self.batch_size = batch_size
        # Fading autoregressive coefficient initialization
        self.rho_fading = config.tf_rng.uniform(
            [batch_size, num_rx, num_tx], minval=0.95, maxval=0.99, dtype=self.rdtype
        )
        # Fading initialization
        self.fading = tf.ones([batch_size, num_rx, num_tx], dtype=self.rdtype)

    def call(self, channel_model):
        """Generate OFDM channel matrix"""

        # Instantiate the OFDM channel generator
        ofdm_channel = GenerateOFDMChannel(channel_model, self.resource_grid)

        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_freq = ofdm_channel(self.batch_size)
        return h_freq

    def update(self, channel_model, h_freq, slot):
        """Update channel matrix every coherence_time slots"""
        # Generate new channel realization
        h_freq_new = self.call(channel_model)

        # Change to new channel every coherence_time slots
        change = tf.cast(tf.math.mod(slot, self.coherence_time) == 0, self.cdtype)
        h_freq = change * h_freq_new + (tf.cast(1, self.cdtype) - change) * h_freq
        return h_freq

    def apply_fading(self, h_freq):
        """Apply fading, modeled as an autoregressive process, to channel matrix"""
        # Multiplicative fading factor evolving via an AR process
        # [batch_size, num_rx, num_tx]
        self.fading = (
            tf.cast(1, self.rdtype)
            - self.rho_fading
            + self.rho_fading * self.fading
            + config.tf_rng.uniform(
                self.fading.shape, minval=-0.1, maxval=0.1, dtype=self.rdtype
            )
        )
        self.fading = tf.maximum(self.fading, tf.cast(0, self.rdtype))
        # [batch_size, num_rx, 1, num_tx, 1, 1, 1]
        fading_expand = insert_dims(self.fading, 1, axis=2)
        fading_expand = insert_dims(fading_expand, 3, axis=4)

        # Channel matrix in the current slot
        h_freq_fading = tf.cast(tf.math.sqrt(fading_expand), self.cdtype) * h_freq
        return h_freq_fading
