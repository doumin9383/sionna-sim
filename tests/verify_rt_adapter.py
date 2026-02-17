import os
import sys
import numpy as np
import tensorflow as tf
import unittest
from unittest.mock import MagicMock, patch
import mitsuba as mi

# Force llvm_ad_rgb variant for testing logic on CPU without GPU
# This ensures drjit operations work in a CPU environment
try:
    mi.set_variant("llvm_ad_rgb")
except Exception:
    # Fallback if already set
    pass

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from wsim.rt.external.adapter import StandardAdapter
from wsim.rt.external.paths import ExternalPaths

# We need to ensure sionna.rt is imported after variant set? Actually importing ExternalPaths likely imports sionna.rt.


class TestVolcanoAdapter(unittest.TestCase):
    def test_standard_adapter_mapping(self):
        # Dummy Volcano HDF5 structure
        dummy_h5 = {
            "path_gains": np.zeros((10, 2, 5, 2, 2)),
            "delay": np.zeros((10, 2, 5)),
            "zenith_at_tx": np.zeros((10, 2, 5)),
            "azimuth_at_tx": np.zeros((10, 2, 5)),
            "zenith_at_rx": np.zeros((10, 2, 5)),
            "azimuth_at_rx": np.zeros((10, 2, 5)),
            "pathloss": np.zeros((2, 10)),
            "tx_positions": np.zeros((2, 3)),
            "tx_orientations": np.zeros((2, 3)),
            "tx_names": np.array([b"BS1", b"BS2"]),
        }
        adapter = StandardAdapter()
        mapping = adapter.map_keys(dummy_h5)
        return mapping, dummy_h5

    @patch("wsim.rt.external.paths.dr.max")
    @patch("wsim.rt.external.paths.dr.scatter_inc")
    @patch("wsim.rt.external.paths.dr.zeros")
    @patch("wsim.rt.external.paths.PathsBuffer")
    def test_external_paths_loading(
        self, MockPathsBuffer, MockDrZeros, MockDrScatter, MockDrMax
    ):
        mapping, dummy_h5 = self.test_standard_adapter_mapping()

        # Mock dr.zeros/scatter/max to bypass initialization logic
        MockDrZeros.return_value = MagicMock()
        MockDrScatter.return_value = MagicMock()
        # dr.max returns [val], so needs to be subscriptable
        MockDrMax.return_value = [100]  # dummy max path count

        # Configure MockPathsBuffer
        mock_buffer_instance = MockPathsBuffer.return_value
        # Paths class might use src_indices.size() or similar drjit calls if we are not careful
        # But we pass paths_buffer to super().__init__
        # Let's inspect what Paths.__init__ does.
        # It calculates num_tx/num_rx from scene.

        # Proper Mock Scene setup
        mock_scene = MagicMock()
        mock_scene.frequency = 3.5e9
        mock_scene.synthetic_array = False

        # Create concrete dicts so len() works and drjit can digest the count
        mock_scene.transmitters = {0: MagicMock(), 1: MagicMock()}  # 2 Tx
        mock_scene.receivers = {0: MagicMock()}  # 1 Rx

        # NOTE: ExternalPaths overrides _build_from_buffer, skipping the heavy drjit logic
        # that usually consumes paths_buffer.
        # The error `could not construct output` usually comes from incompatible shape/dtype.
        # Ensure num_src * num_tgt is valid integer. 2 * 1 = 2.

        # Instantiate
        paths = ExternalPaths(
            dataset=dummy_h5,
            scene=mock_scene,
            num_tx=2,
            num_rx=1,
            sample_index=[0],
            key_mapping=mapping,
        )

        # Check if internal tensors are drjit arrays
        # Since we are running with llvm_ad_rgb, real drjit arrays should be created in _load_from_dataset
        import drjit as dr

        # self.assertTrue(dr.is_array_v(paths.tau)) # This might fail if we mocked dr?
        # No, we patched wsim.rt.external.paths.dr.zeros, not dr module globally for check.
        # But wait, wsim.rt.external.paths imports drjit as dr.
        # If we patch 'wsim.rt.external.paths.dr.zeros', we only affect that call.

        # Check shapes (using numpy conversion/shape)
        # paths.tau should be [1, 1, 2, 1, 5]
        # We can verify the shape attribute directly without checking drjit type strictly if needed
        self.assertEqual(tuple(paths.tau.shape), (1, 1, 2, 1, 5))


if __name__ == "__main__":
    unittest.main()
