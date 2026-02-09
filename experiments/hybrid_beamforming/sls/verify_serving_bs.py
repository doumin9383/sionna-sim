import sys
import os
import tensorflow as tf
import numpy as np

# Adjust path to find modules
sys.path.append(os.getcwd())

from experiments.hybrid_beamforming.sls.simulator import HybridSystemSimulator
from experiments.hybrid_beamforming.sls.my_configs import HybridSLSConfig


def verify_serving_bs_association():
    print("Verifying Serving BS Association...")

    # Create Config
    config = HybridSLSConfig()
    config.num_rings = 1
    config.num_ut_per_sector = 2
    config.batch_size = 1
    config.num_neighbors = 5

    print(
        f"Config: num_rings={config.num_rings}, num_ut_per_sector={config.num_ut_per_sector}"
    )

    # Initialize Simulator
    try:
        sim = HybridSystemSimulator(config)
    except Exception as e:
        print(f"Error initializing simulator: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"Simulator initialized. Num BS: {sim.num_bs}, Num UT: {sim.num_ut}")

    # Trigger topology setup
    print("Setting up topology...")
    sim._setup_topology(config.num_rings, config.min_bs_ut_dist, config.max_bs_ut_dist)

    # Check Serving BS IDs
    serving_bs_ids = sim.serving_bs_ids.numpy()
    neighbor_indices = sim.neighbor_indices.numpy()

    print("\n--- Checking Association ---")

    # neighbor_indices: [batch, num_ut, num_neighbors]
    # serving_bs_ids: [batch, num_ut]

    # Check if neighbor_indices[:,:,0] == serving_bs_ids
    serving_bs_from_neighbor = neighbor_indices[:, :, 0]

    mismatch_count = np.sum(serving_bs_from_neighbor != serving_bs_ids)

    if mismatch_count == 0:
        print("SUCCESS: All UEs have correct Serving BS at neighbor index 0.")
    else:
        print(f"FAILURE: {mismatch_count} mismatches found!")

        # Debug info
        rows, cols = np.where(serving_bs_from_neighbor != serving_bs_ids)
        for r, c in zip(rows[:5], cols[:5]):
            expected = serving_bs_ids[r, c]
            actual = serving_bs_from_neighbor[r, c]
            print(f"Batch {r}, UE {c}: Expected BS {expected}, Got BS {actual}")

    # Also check if other neighbors are valid (simple check)
    print("\nSample Neighbor Indices (UE 0):")
    print(neighbor_indices[0, 0, :])
    print(f"Expected Serving BS: {serving_bs_ids[0, 0]}")


if __name__ == "__main__":
    verify_serving_bs_association()
