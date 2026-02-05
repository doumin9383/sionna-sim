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


def init_result_history(batch_size, num_slots, num_bs, num_ut_per_sector):
    """Initialize dictionary containing history of results"""
    hist = {}
    for key in [
        "pathloss_serving_cell",
        "tx_power",
        "olla_offset",
        "sinr_eff",
        "pf_metric",
        "num_decoded_bits",
        "mcs_index",
        "harq",
        "num_allocated_re",
    ]:
        hist[key] = tf.TensorArray(
            size=num_slots,
            element_shape=[batch_size, num_bs, num_ut_per_sector],
            dtype=tf.float32,
        )
    return hist


def record_results(
    hist,
    slot,
    sim_failed=False,
    pathloss_serving_cell=None,
    num_allocated_re=None,
    tx_power_per_ut=None,
    num_decoded_bits=None,
    mcs_index=None,
    harq_feedback=None,
    olla_offset=None,
    sinr_eff=None,
    pf_metric=None,
    shape=None,
):
    """Record results of last slot"""
    if not sim_failed:
        for key, value in zip(
            [
                "pathloss_serving_cell",
                "olla_offset",
                "sinr_eff",
                "num_allocated_re",
                "tx_power",
                "num_decoded_bits",
                "mcs_index",
                "harq",
            ],
            [
                pathloss_serving_cell,
                olla_offset,
                sinr_eff,
                num_allocated_re,
                tx_power_per_ut,
                num_decoded_bits,
                mcs_index,
                harq_feedback,
            ],
        ):
            hist[key] = hist[key].write(slot, tf.cast(value, tf.float32))
        # Average PF metric across resources
        hist["pf_metric"] = hist["pf_metric"].write(
            slot, tf.reduce_mean(pf_metric, axis=[-2, -3])
        )
    else:
        nan_tensor = tf.cast(tf.fill(shape, float("nan")), dtype=tf.float32)
        for key in hist:
            hist[key] = hist[key].write(slot, nan_tensor)
    return hist


def clean_hist(hist, batch=0):
    """Extract batch, convert to Numpy, and mask metrics when user is not
    scheduled"""
    # Extract batch and convert to Numpy
    for key in hist:
        try:
            # [num_slots, num_bs, num_ut_per_sector]
            hist[key] = hist[key].numpy()[:, batch, :, :]
        except:
            pass

    # Mask metrics when user is not scheduled
    hist["mcs_index"] = np.where(hist["harq"] == -1, np.nan, hist["mcs_index"])
    hist["sinr_eff"] = np.where(hist["harq"] == -1, np.nan, hist["sinr_eff"])
    hist["tx_power"] = np.where(hist["harq"] == -1, np.nan, hist["tx_power"])
    hist["num_allocated_re"] = np.where(hist["harq"] == -1, 0, hist["num_allocated_re"])
    hist["harq"] = np.where(hist["harq"] == -1, np.nan, hist["harq"])
    return hist
