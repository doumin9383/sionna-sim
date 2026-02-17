#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import mitsuba as mi
import drjit as dr
import zarr
import numpy as np

from sionna.rt import Paths, Scene
from sionna.rt.path_solvers.paths_buffer import PathsBuffer


class ExternalPaths(Paths):
    """
    Extensions of the Sionna Paths class to support loading from external Zarr stores.

    This class bypasses the standard ray tracing pipeline and directly injects
    pre-computed path data (gain, delay, angles) into the internal tensors.
    """

    def __init__(
        self,
        dataset: any,
        scene: Scene,
        num_tx: int,
        num_rx: int,
        num_tx_ant: int = 1,
        num_rx_ant: int = 1,
        sample_index: int = None,
        key_mapping: dict = None,
    ):
        """
        Initializes the ExternalPaths object by loading data from a dataset (Zarr/HDF5).

        Args:
            dataset (any): Open dataset or mapping containing the ray tracing results.
            scene (Scene): The Sionna scene object (needed for frequency/wavelength).
            num_tx (int): Number of transmitters.
            num_rx (int): Number of receivers.
            num_tx_ant (int, optional): Number of antennas per transmitter. Defaults to 1.
            num_rx_ant (int, optional): Number of antennas per receiver. Defaults to 1.
            sample_index (int, optional): Index of the sample/scenario to load if the dataset
                                          contains multiple samples (first dimension). Defaults to None.
            key_mapping (dict, optional): Mapping from standard keys to dataset keys.
        """

        # 1. Bypass standard initialization with a dummy buffer
        dummy_buffer = PathsBuffer(buffer_size=1, max_depth=1, diffraction=False)

        # Pre-initialize attributes required by _load_from_dataset (called via super().__init__ -> _build_from_buffer)
        self._dataset = dataset
        self._num_tx = num_tx
        self._num_rx = num_rx
        self._sample_index = sample_index
        self._key_mapping = key_mapping or {}

        # Call parent generic init with minimal dummy values
        super().__init__(
            scene=scene,
            src_positions=mi.Point3f(0, 0, 0),  # Dummy
            tgt_positions=mi.Point3f(0, 0, 0),  # Dummy
            tx_velocities=mi.Vector3f(0, 0, 0),  # Dummy
            rx_velocities=mi.Vector3f(0, 0, 0),  # Dummy
            synthetic_array=False,  # Set to False to trigger _build_from_buffer (which we override)
            paths_buffer=dummy_buffer,
            rel_ant_positions_tx=None,
            rel_ant_positions_rx=None,
        )

    def _load_from_dataset(self):
        """
        Loads the path data from the dataset and populates the internal DrJit tensors.
        """
        store = self._dataset

        try:

            def get_data(standard_key):
                # Resolve key from mapping, default to standard_key if not found
                key = self._key_mapping.get(standard_key, standard_key)
                if key not in store:
                    return None
                data = store[key]
                if self._sample_index is not None:
                    # dataset[indices] in h5py/zarr returns the sliced array
                    data = data[self._sample_index]
                # Ensure data is a numpy array (important for TF/DrJit from HDF5)
                return np.array(data)

            # Load delay first (required)
            delay = get_data("delay")
            if delay is None:
                raise KeyError("Dataset must contain 'delay'")

            # Helper to reshape 3D [RX, TX, Paths] to 5D [RX, 1, TX, 1, Paths]
            def reshape_to_5d(tensor_data):
                if len(tensor_data.shape) == 3:
                    return tensor_data[:, np.newaxis, :, np.newaxis, :]
                return tensor_data

            delay = reshape_to_5d(delay)
            flat_delay = delay.flatten()
            self._tau = (
                dr.cuda.Float(flat_delay)
                if mi.variant().startswith("cuda")
                else dr.llvm.Float(flat_delay)
            )
            self._tau = dr.reshape(mi.TensorXf, self._tau, delay.shape)

            # Load path gains
            # 'path_gains' is used as the standard key for both polarized and scalar
            # We inspect the shape to decide
            path_gains = get_data(
                "path_gains"
            )  # Usually mapped from 'path_gains' or 'path_gain'

            if path_gains is not None:
                if path_gains.ndim >= 5 or (
                    path_gains.ndim >= 2 and path_gains.shape[-2:] == (2, 2)
                ):
                    # Polarized: [..., Paths, 2, 2]
                    # Reshape to [num_rx, 1, num_tx, 1, num_paths, 2, 2]
                    if len(path_gains.shape) == 5:
                        path_gains = path_gains[:, np.newaxis, :, np.newaxis, :, :, :]

                    flat_real = np.real(path_gains).flatten()
                    flat_imag = np.imag(path_gains).flatten()

                    self._a_real = (
                        dr.cuda.Float(flat_real)
                        if mi.variant().startswith("cuda")
                        else dr.llvm.Float(flat_real)
                    )
                    self._a_real = dr.reshape(
                        mi.TensorXf, self._a_real, path_gains.shape
                    )

                    self._a_imag = (
                        dr.cuda.Float(flat_imag)
                        if mi.variant().startswith("cuda")
                        else dr.llvm.Float(flat_imag)
                    )
                    self._a_imag = dr.reshape(
                        mi.TensorXf, self._a_imag, path_gains.shape
                    )

                else:
                    # Scalar: [..., Paths]
                    # path_gain is assumed to be power linear -> amplitude = sqrt(power)
                    path_gain = reshape_to_5d(path_gains)
                    amplitude = np.sqrt(path_gain)

                    # If phase is available
                    phase = get_data("phase")
                    if phase is not None:
                        phase = reshape_to_5d(phase)
                        # Compute complex amplitude
                        a_complex = amplitude * (np.cos(phase) + 1j * np.sin(phase))

                        flat_real = np.real(a_complex).flatten()
                        flat_imag = np.imag(a_complex).flatten()

                        self._a_real = (
                            dr.cuda.Float(flat_real)
                            if mi.variant().startswith("cuda")
                            else dr.llvm.Float(flat_real)
                        )
                        self._a_real = dr.reshape(
                            mi.TensorXf, self._a_real, path_gain.shape
                        )

                        self._a_imag = (
                            dr.cuda.Float(flat_imag)
                            if mi.variant().startswith("cuda")
                            else dr.llvm.Float(flat_imag)
                        )
                        self._a_imag = dr.reshape(
                            mi.TensorXf, self._a_imag, path_gain.shape
                        )
                    else:
                        # No phase, real only
                        flat_amp = amplitude.flatten()
                        self._a_real = (
                            dr.cuda.Float(flat_amp)
                            if mi.variant().startswith("cuda")
                            else dr.llvm.Float(flat_amp)
                        )
                        self._a_real = dr.reshape(
                            mi.TensorXf, self._a_real, path_gain.shape
                        )
                        self._a_imag = dr.zeros(mi.TensorXf, self._a_real.shape)

            else:
                # Fallback: try explicit 'path_gain' if 'path_gains' failed (less likely with adapter)
                raise KeyError(
                    "Dataset must contain 'path_gains' (polarized) or 'path_gain' (scalar)."
                )

            # Angles
            def load_angle(key):
                val = get_data(key)
                if val is not None:
                    val = reshape_to_5d(val)
                    flat_val = val.flatten()
                    t = (
                        dr.cuda.Float(flat_val)
                        if mi.variant().startswith("cuda")
                        else dr.llvm.Float(flat_val)
                    )
                    return dr.reshape(mi.TensorXf, t, val.shape)
                return dr.zeros(mi.TensorXf, self._tau.shape)

            self._theta_t = load_angle("zenith_at_tx")
            self._phi_t = load_angle("azimuth_at_tx")
            self._theta_r = load_angle("zenith_at_rx")
            self._phi_r = load_angle("azimuth_at_rx")

            # Doppler (optional)
            self._doppler = load_angle("doppler")

            # --- LSP Loading/Calculation ---
            # SLS needs explicit LSPs (Pathloss, Shadowing, K-Factor)
            self._lsps = {}

            # 1. Try to load explicit LSPs from dataset
            for key in ["pathloss", "shadow_fading", "k_factor"]:
                data = get_data(key)
                if data is not None:
                    self._lsps[key] = data

            # 2. Fallback for pathloss if missing
            if "pathloss" not in self._lsps:
                # Try getting path_gains via the same logic as above
                pg = get_data("path_gains")
                if pg is not None:
                    if pg.ndim >= 5 or (pg.ndim >= 2 and pg.shape[-2:] == (2, 2)):
                        # path_gains: [RX, TX, Paths, 2, 2]
                        # Total Rx power (normalized to 1W Tx) = Sum(|path_gains|^2)
                        rx_power_linear = np.sum(np.abs(pg) ** 2, axis=(-3, -2, -1))
                    else:
                        # path_gain: [RX, TX, Paths]
                        rx_power_linear = np.sum(pg, axis=-1)
                else:
                    rx_power_linear = None

                if rx_power_linear is not None:
                    # Pathloss [dB] = Ptx[dBm] - Prx[dBm] = 30 - 10*log10(Prx)
                    # Use a small epsilon to avoid log(0)
                    pl_db = 30.0 - 10.0 * np.log10(np.maximum(rx_power_linear, 1e-15))
                    self._lsps["pathloss"] = pl_db

            # Set valid flag
            self._valid = dr.full(mi.TensorXb, True, self._tau.shape)
            self._paths_components_built = False

        except Exception as e:
            raise RuntimeError(f"Failed to load external paths from dataset: {e}")

    @property
    def lsps(self) -> dict:
        """Returns the dictionary of LSPs (Pathloss, Shadowing, K-Factor, etc.)"""
        return self._lsps

    def _build_from_buffer(self):
        """
        Bypass the standard construction of tensors from the paths buffer.
        """
        self._load_from_dataset()

    def _fuse_pattern_array_dims(self):
        """
        Bypass standard fusion.
        External data is assumed to be loaded in the final fused shape
        [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] directly.
        """
        pass
