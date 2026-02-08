import pickle
import numpy as np
import tensorflow as tf
import os

path = "experiments/hybrid_beamforming/sls/results/history.pkl"

if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

with open(path, "rb") as f:
    hist = pickle.load(f)

print("--- History Inspection ---")
for key, val in hist.items():
    if isinstance(val, tf.Tensor):
        val = val.numpy()

    print(f"{key}: shape={val.shape}, type={val.dtype}")
    if np.size(val) > 0:
        print(f"  Mean: {np.mean(val):.4e}")
        print(f"  Max:  {np.max(val):.4e}")
        print(f"  Min:  {np.min(val):.4e}")
        # Check for zeros
        if np.all(val == 0):
            print("  WARNING: ALL ZEROS")

print("\n--- Detailed SINR (First Slot, First User) ---")
if "sinr_eff" in hist:
    # [slots, batch, bs, ut_per_sector]
    sinr = hist["sinr_eff"]
    print(f"SINR shape: {sinr.shape}")
    print(f"Values: {sinr[0, 0, :, 0]}")  # First slot, batch 0, all BS, user 0

print("\n--- Detailed MCS (First Slot, First User) ---")
if "mcs_index" in hist:
    mcs = hist["mcs_index"]
    print(f"MCS shape: {mcs.shape}")
    print(f"Values: {mcs[0, 0, :, 0]}")

print("\n--- Detailed Throughput (First Slot, First User) ---")
if "num_decoded_bits" in hist:
    # Actually this is Throughput [bps]
    tput = hist["num_decoded_bits"]
    print(f"Throughput shape: {tput.shape}")
    print(f"Values: {tput[0, 0, :, 0]}")
