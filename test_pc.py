import tensorflow as tf
from sionna.sys import open_loop_uplink_power_control
from sionna.phy.utils import db_to_lin, watt_to_dbm
import pprint

pl_db = tf.constant([135.14912, 114.36259, -10.0]) # Add a negative pathloss to test
num_rbs = tf.constant([273.0, 273.0, 273.0])
path_loss_lin = db_to_lin(pl_db)
num_subcarriers = num_rbs * 12.0
p_cmax = tf.constant([23.0, 23.0, 23.0])

p_tx_watt = open_loop_uplink_power_control(
    pathloss=path_loss_lin,
    num_allocated_subcarriers=num_subcarriers,
    alpha=0.8,
    p0_dbm=-80.0,
    ut_max_power_dbm=p_cmax,
)

print("pl_db:", pl_db.numpy())
print("p_tx_watt:", p_tx_watt.numpy())
print("p_tx_dbm:", watt_to_dbm(p_tx_watt).numpy())
