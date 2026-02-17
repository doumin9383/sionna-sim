import os
from sionna.rt import Scene
from wsim.rt.scene_setup import setup_scene_from_hdf5
import numpy as np


def test_scene_setup():
    filename = "dummy_rt_data.h5"
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run create_dummy_hdf5.py first.")
        return

    # Create empty scene
    scene = Scene()

    # Run setup
    print(f"Setting up scene from {filename}...")
    setup_scene_from_hdf5(scene, filename)

    # Verification
    print("\nVerification Results:")

    # 1. Origin
    # Note: scene.origin is hard to check directly as property?, actually usually accessible.
    # But let's check transmitters relative position.

    # 2. Transmitters
    num_tx = len(scene.transmitters)
    print(f"  Number of Transmitters: {num_tx}")

    if num_tx == 0:
        print("[FAIL] No transmitters added.")
        return

    # Check first transmitter
    tx_names = list(scene.transmitters.keys())
    tx0 = scene.transmitters[tx_names[0]]
    print(f"  First TX Name: {tx_names[0]}")
    print(f"  First TX Position (Local): {tx0.position.numpy()}")
    print(f"  First TX Orientation: {tx0.orientation.numpy()}")

    # Expected logic: Local = Global - Origin
    import h5py

    with h5py.File(filename, "r") as f:
        origin = f.attrs["origin_utm"]
        tx_pos_global = f["tx_positions"][0]
        expected_local = tx_pos_global - origin

        # Check proximity
        if np.allclose(
            tx0.position.numpy().flatten(), expected_local.flatten(), atol=1e-4
        ):
            print("[OK] Position matches expected local coordinates.")
        else:
            print(
                f"[FAIL] Position mismatch. Expected {expected_local}, Got {tx0.position.numpy()}"
            )


if __name__ == "__main__":
    test_scene_setup()
