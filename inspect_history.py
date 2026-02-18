import pickle
import numpy as np
import tensorflow as tf

history_path = "experiments/hybrid_beamforming/sls/results/history.pkl"

with open(history_path, "rb") as f:
    history = pickle.load(f)

print("Keys:", history.keys())

if "mpr_db" in history:
    mpr = history["mpr_db"]
    print(f"MPR Shape: {mpr.shape}")
    print(f"MPR Mean: {np.mean(mpr)}")
    print(f"MPR Min: {np.min(mpr)}")
    print(f"MPR Max: {np.max(mpr)}")
    print(f"MPR Sample: {mpr.flatten()[:10]}")

if "p_cmax_dbm" in history:
    p_cmax = history["p_cmax_dbm"]
    print(f"P_cmax Mean: {np.mean(p_cmax)}")
    print(f"P_cmax Sample: {p_cmax.flatten()[:10]}")

if "tx_power" in history:
    tx_p = history["tx_power"]  # In Watts?
    tx_p_dbm = 10 * np.log10(np.maximum(tx_p, 1e-20)) + 30
    print(f"Tx Power (dBm) Max: {np.max(tx_p_dbm)}")
    print(f"Tx Power (dBm) Sample: {tx_p_dbm.flatten()[:10]}")
