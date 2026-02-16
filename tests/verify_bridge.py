import os
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock
from wsim.sls.components.channel_adapters import MeshToSLSAdapter


def verify_bridge():
    print("Starting verification of SLS-RT Bridge (with mocking)...")

    # 1. Mock Loader and Paths
    loader = MagicMock()
    paths = MagicMock()

    # Setup Paths attributes (mocking tf/numpy behavior)
    paths.tau = tf.constant(np.random.rand(10, 1, 2, 1, 5), dtype=tf.float32)
    paths.a = tf.constant(np.random.rand(10, 1, 2, 1, 5), dtype=tf.complex64)
    paths.phi_r = tf.constant(np.random.rand(10, 1, 2, 1, 5), dtype=tf.float32)
    paths.phi_t = tf.constant(np.random.rand(10, 1, 2, 1, 5), dtype=tf.float32)
    paths.theta_r = tf.constant(np.random.rand(10, 1, 2, 1, 5), dtype=tf.float32)
    paths.theta_t = tf.constant(np.random.rand(10, 1, 2, 1, 5), dtype=tf.float32)

    # Extension: lsps
    paths.lsps = {
        "pathloss": np.random.rand(10, 2),
        "shadow_fading": np.random.rand(10, 2),
    }

    loader.get_paths.return_value = paths

    # 2. Setup Adapter
    adapter = MeshToSLSAdapter(loader)

    # 3. Test coordinate/index access (logic is in adapter)
    print("Testing adapter dictionary conversion...")
    ut_info = [[0.0, 0.0, 0.0]]
    params = adapter.get_channel_params(ut_info)

    # Check keys
    required_keys = [
        "delays",
        "powers",
        "pathloss",
        "aoa",
        "aod",
        "zoa",
        "zod",
        "shadow_fading",
    ]
    for key in required_keys:
        assert key in params, f"Missing key: {key}"

    # Check shapes [Batch, RX, TX, Paths] or [Batch, TX, RX] for LSPs
    assert params["delays"].shape == (1, 10, 2, 5)
    assert params["powers"].shape == (1, 10, 2, 5)
    assert params["pathloss"].shape == (1, 2, 10)  # [Batch, TX, RX]

    print("Verification OK!")


if __name__ == "__main__":
    verify_bridge()
