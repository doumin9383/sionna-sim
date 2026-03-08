import tensorflow as tf
import numpy as np
from experiments.hybrid_beamforming.sls.configs import SLSConfig
from experiments.hybrid_beamforming.sls.components.scheduler import PFScheduler

config = SLSConfig()
config.waveform = "CP-OFDM"

scheduler = PFScheduler(config, num_ut=21, num_rb=66)
n_req = tf.constant([[64] * 21], dtype=tf.int32)
max_thp = tf.constant([[0.0, 73063.4, 914.5] + [100.0]*18], dtype=tf.float32)
mcs_opt = tf.constant([[3] * 21], dtype=tf.int32)
rank_opt = tf.constant([[1] * 21], dtype=tf.int32)

pre_allocation_results = {
    "n_req": n_req,
    "max_thp": max_thp,
    "mcs_opt": mcs_opt,
    "rank_opt": rank_opt
}

res = scheduler.schedule(pre_allocation_results)
print("scheduled_rbs:", res["scheduled_rbs"].numpy())
print("sum alloc mask:", tf.reduce_sum(tf.cast(res["allocation_mask"], tf.float32)).numpy())

# test with DFT-s-OFDM
config.waveform = "DFT-s-OFDM"
scheduler = PFScheduler(config, num_ut=21, num_rb=66)
res2 = scheduler.schedule(pre_allocation_results)
print("DFT scheduled_rbs:", res2["scheduled_rbs"].numpy())

