import tensorflow as tf
import numpy as np
from typing import Union, List
from wsim.rt.external.loaders import ExternalLoaderBase


class MeshToSLSAdapter:
    """
    Adapter that wraps a MeshBasedLoader (or any ExternalLoaderBase returning Extensions of Paths)
    and provides a channel parameter dictionary compatible with SLS HybridChannelInterface.
    """

    def __init__(self, loader: ExternalLoaderBase):
        self._loader = loader

    def get_channel_params(
        self, ut_info: Union[np.ndarray, tf.Tensor, List[int]]
    ) -> dict:
        """
        Retrieves channel parameters and LSPs for the given UT coordinates or mesh indices.

        Args:
            ut_info: [num_rx, 3] coordinates or [num_rx] mesh indices.

        Returns:
            dict: Dictionary containing 'delays', 'powers', 'pathloss', 'aoa', etc.
        """
        # 1. Get Paths object from loader
        # MeshBasedLoader.get_paths now supports both coordinates and indices
        paths = self._loader.get_paths(ut_info)

        # 2. Extract Ray information
        # Sionna Paths properties are [batch, rx, rx_ant, tx, tx_ant, paths, ...]
        # SLS expects roughly [1, rx, tx, paths] for rays

        # Helper to convert DrJit/TF tensor and remove antenna dimensions if they are 1
        def prepare_tensor(t):
            if hasattr(t, "numpy"):
                t_np = t.numpy()
            else:
                t_np = np.array(t)
            # Paths tensors are usually 5D or 7D.
            # We want to squeeze antenna dimensions if they are 1.
            # Paths: [RX, 1, TX, 1, Paths] (from ExternalPaths)
            # To match generate_dataset: [1, RX, TX, Paths]
            if t_np.ndim == 5:
                # [RX, 1, TX, 1, Paths] -> [RX, TX, Paths]
                t_np = np.squeeze(t_np, axis=(1, 3))
                # -> [1, RX, TX, Paths]
                t_np = t_np[np.newaxis, ...]
            return tf.constant(t_np, dtype=tf.float32)

        # Map Paths attributes to SLS dictionary keys
        # tau: [RX, 1, TX, 1, Paths]
        # a: [RX, 1, TX, 1, Paths, 2, 2] or [RX, 1, TX, 1, Paths] (if scalar)

        # Calculate powers from complex amplitude 'a'
        # a is [RX, 1, TX, 1, Paths, (2, 2)]
        a_complex = paths.a.numpy()
        if a_complex.ndim == 7:  # Polarized [RX, 1, TX, 1, Paths, 2, 2]
            powers = np.sum(np.abs(a_complex) ** 2, axis=(-2, -1))
        else:  # Scalar [RX, 1, TX, 1, Paths]
            powers = np.abs(a_complex) ** 2

        results = {
            "delays": prepare_tensor(paths.tau),
            "powers": prepare_tensor(powers),
            "aoa": prepare_tensor(paths.phi_r),
            "aod": prepare_tensor(paths.phi_t),
            "zoa": prepare_tensor(paths.theta_r),
            "zod": prepare_tensor(paths.theta_t),
        }

        # 3. Inject LSPs (Pathloss, Shadowing, K-Factor)
        if hasattr(paths, "lsps"):
            for key, val in paths.lsps.items():
                if isinstance(val, (np.ndarray, tf.Tensor)):
                    # LSPs: [RX, TX] -> [1, TX, RX] (to match ExternalChannelLoader/generate_dataset)
                    # generate_dataset saves LSPs as [Batch, TX, RX]
                    # Wait, lets check generate_dataset.py line 205: grp_rays.create_dataset(key, data=data)
                    # data was concatenated along axis 2 (UT), so [Batch, BS, UT]
                    if val.ndim == 2:  # [RX, TX]
                        val = np.transpose(val)  # [TX, RX]
                        val = val[np.newaxis, ...]  # [1, TX, RX]
                    results[key] = tf.constant(val, dtype=tf.float32)

        return results
