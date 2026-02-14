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
        zarr_path: str,
        scene: Scene,
        num_tx: int,
        num_rx: int,
        num_tx_ant: int = 1,
        num_rx_ant: int = 1,
        sample_index: int = None,
    ):
        """
        Initializes the ExternalPaths object by loading data from a Zarr store.

        Args:
            zarr_path (str): Path to the Zarr store containing the ray tracing results.
            scene (Scene): The Sionna scene object (needed for frequency/wavelength).
            num_tx (int): Number of transmitters.
            num_rx (int): Number of receivers.
            num_tx_ant (int, optional): Number of antennas per transmitter. Defaults to 1.
            num_rx_ant (int, optional): Number of antennas per receiver. Defaults to 1.
            sample_index (int, optional): Index of the sample/scenario to load if the Zarr store
                                          contains multiple samples (first dimension). Defaults to None (load all/squeeze).
        """

        # 1. Bypass standard initialization with a dummy buffer
        # We create a minimal buffer to satisfy the parent constructor
        # but we will immediately overwrite the internal state.
        dummy_buffer = PathsBuffer(buffer_size=1, max_depth=1, diffraction=False)

        # Pre-initialize attributes required by _load_from_zarr (called via super().__init__ -> _build_from_buffer)
        self._zarr_path = zarr_path
        self._num_tx = num_tx
        self._num_rx = num_rx
        self._sample_index = sample_index

        # Call parent generic init with minimal dummy values
        # Note: We assume synthetic_array=False for direct injection for now,
        # as we are handling full path data unless specified otherwise.
        super().__init__(
            scene=scene,
            src_positions=mi.Point3f(0, 0, 0),  # Dummy
            tgt_positions=mi.Point3f(0, 0, 0),  # Dummy
            tx_velocities=mi.Vector3f(0, 0, 0),  # Dummy
            rx_velocities=mi.Vector3f(0, 0, 0),  # Dummy
            synthetic_array=False,
            paths_buffer=dummy_buffer,
            rel_ant_positions_tx=None,
            rel_ant_positions_rx=None,
        )

    def _load_from_zarr(self):
        """
        Loads the path data from Zarr and populates the internal DrJit tensors.
        """
        store = zarr.open(self._zarr_path, mode="r")

        try:
            # Helper to load and convert to DrJit tensor with optional slicing
            def load_tensor(key, default_val=0.0):
                if key in store:
                    data = store[key]
                    if (
                        self._sample_index is not None and data.shape[0] > 1
                    ):  # Assuming dim 0 is sample
                        # Check if data has enough dimensions to be sliced
                        if len(data.shape) > 0:
                            data = data[self._sample_index]

                    data = np.array(data)
                    return (
                        dr.cuda.Float(data)
                        if mi.variant().startswith("cuda")
                        else dr.llvm.Float(data)
                    )
                return dr.full(
                    mi.TensorXf, default_val, [1]
                )  # Fallback, likely needs better shape handling

            # Load core path data -> These keys match StandardAdapter
            # We load as numpy first to check shapes easily
            if "path_gain" in store:
                path_gain = store["path_gain"]
            else:
                raise KeyError("Zarr store must contain 'path_gain'")

            if "delay" in store:
                delay = store["delay"]
            else:
                raise KeyError("Zarr store must contain 'delay'")

            # Slice if needed
            if self._sample_index is not None:
                path_gain = path_gain[self._sample_index]
                delay = delay[self._sample_index]

            path_gain = np.array(path_gain)
            delay = np.array(delay)

            # Expected shape assumption: [num_rx, num_tx, num_paths] (SISO default)
            # We need to reshape to [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
            # Assuming 1 antenna for now as per defaults in standard HDF5 ingestion

            # Helper to reshape 3D [RX, TX, Paths] to 5D [RX, 1, TX, 1, Paths]
            def reshape_to_5d(tensor_data):
                if len(tensor_data.shape) == 3:
                    return tensor_data[:, np.newaxis, :, np.newaxis, :]
                return tensor_data

            path_gain = reshape_to_5d(path_gain)
            delay = reshape_to_5d(delay)

            # Populating internal state
            # DrJit often needs flat input for efficient copy from numpy
            flat_delay = delay.flatten()
            self._tau = (
                dr.cuda.Float(flat_delay)
                if mi.variant().startswith("cuda")
                else dr.llvm.Float(flat_delay)
            )
            self._tau = dr.reshape(mi.TensorXf, self._tau, delay.shape)

            # Assuming path_gain is power linear -> amplitude = sqrt(power)
            amplitude = np.sqrt(path_gain)

            # If phase is available in Zarr
            if "phase" in store:
                phase = store["phase"]
                if self._sample_index is not None:
                    phase = phase[self._sample_index]
                phase = np.array(phase)
                phase = reshape_to_5d(phase)

                # Compute complex amplitude in numpy to avoid repeated flat/reshape for cos/sin
                a_complex = amplitude * (np.cos(phase) + 1j * np.sin(phase))

                flat_real = np.real(a_complex).flatten()
                flat_imag = np.imag(a_complex).flatten()

                self._a_real = (
                    dr.cuda.Float(flat_real)
                    if mi.variant().startswith("cuda")
                    else dr.llvm.Float(flat_real)
                )
                self._a_real = dr.reshape(mi.TensorXf, self._a_real, path_gain.shape)

                self._a_imag = (
                    dr.cuda.Float(flat_imag)
                    if mi.variant().startswith("cuda")
                    else dr.llvm.Float(flat_imag)
                )
                self._a_imag = dr.reshape(mi.TensorXf, self._a_imag, path_gain.shape)
            else:
                flat_amp = amplitude.flatten()
                self._a_real = (
                    dr.cuda.Float(flat_amp)
                    if mi.variant().startswith("cuda")
                    else dr.llvm.Float(flat_amp)
                )
                self._a_real = dr.reshape(mi.TensorXf, self._a_real, path_gain.shape)
                self._a_imag = dr.zeros(mi.TensorXf, self._a_real.shape)

            # Angles
            def load_angle(key):
                if key in store:
                    val = store[key]
                    if self._sample_index is not None:
                        val = val[self._sample_index]
                    val = np.array(val)
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
            self._doppler = load_angle(
                "doppler"
            )  # Assuming scalar doppler per path, reuse load_angle/reshape logic

            # Set valid flag
            # Assume all loaded paths are valid
            self._valid = dr.full(mi.TensorXb, True, self._tau.shape)

            self._paths_components_built = False  # We don't have interaction/shape info

        except Exception as e:
            raise RuntimeError(
                f"Failed to load external paths from {self._zarr_path}: {e}"
            )

    def _build_from_buffer(self):
        """
        Bypass the standard construction of tensors from the paths buffer.
        Instead, we trigger the Zarr loader here.
        """
        self._load_from_zarr()

    def _fuse_pattern_array_dims(self):
        """
        Bypass standard fusion.
        External data is assumed to be loaded in the final fused shape
        [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] directly.
        """
        pass
