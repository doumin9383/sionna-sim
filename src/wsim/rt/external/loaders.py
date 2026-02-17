#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import abc
import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree
from typing import Union, Tuple, List, Optional
import zarr
import mitsuba as mi

from sionna.rt import Paths, Scene
from .paths import ExternalPaths
from ...common.geo import CoordinateSystem
from .adapter import BaseAdapter, StandardAdapter


class ExternalLoaderBase(abc.ABC):
    """
    Abstract base class for path loaders.
    """

    @abc.abstractmethod
    def get_paths(self, ut_coordinates_local: Union[np.ndarray, tf.Tensor]) -> Paths:
        """
        Retrieves ray tracing paths for the given local coordinates.

        Args:
            ut_coordinates_local (Union[np.ndarray, tf.Tensor]):
                User terminal coordinates in the local simulation frame [num_rx, 3].

        Returns:
            Paths: A Sionna Paths object.
        """
        pass


import h5py
import os


class MeshBasedLoader(ExternalLoaderBase):
    """
    Loads pre-computed ray tracing data from a mesh grid (Zarr/HDF5).

    Uses KDTree for nearest neighbor search between query points and mesh points.
    """

    def __init__(
        self,
        file_path: str,
        scene: Scene,
        use_3d_search: bool = False,
        adapter: BaseAdapter = None,
    ):
        """
        Args:
            file_path (str): Path to the Zarr store or HDF5 file.
            scene (Scene): The Sionna scene context.
            use_3d_search (bool): Whether to use (x, y, z) for nearest neighbor search.
                                  If False, only (x, y) is used.
            adapter (BaseAdapter, optional): Adapter to map HDF5 keys to standard keys.
                                             Defaults to StandardAdapter.
        """
        self._file_path = file_path
        self._scene = scene
        self._use_3d_search = use_3d_search
        self._adapter = adapter or StandardAdapter()

        # Open store based on extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() in [".h5", ".hdf5"]:
            self._dataset = h5py.File(file_path, "r")
        else:
            # Default to zarr
            self._dataset = zarr.open(file_path, mode="r")

        # Map keys using the adapter
        self._key_mapping = self._adapter.map_keys(self._dataset)

        # Initialize CoordinateSystem from metadata
        # Both h5py and zarr support .attrs (dictionary-like)
        origin_utm = self._dataset.attrs.get("origin_utm", (0.0, 0.0, 0.0))
        self._geo = CoordinateSystem(origin_utm)

        # Load mesh coordinates (UTM) and convert to local
        # mesh_coordinates is fundamental, so we look for it directly or via adapter if added later
        # For now, we assume it's "mesh_coordinates" as per spec
        if "mesh_coordinates" not in self._dataset:
            raise KeyError(f"Dataset at {file_path} must contain 'mesh_coordinates'")

        # mesh_coordinates is usually small compared to path gains, so we read it all
        mesh_utm = np.array(self._dataset["mesh_coordinates"])
        self._mesh_local = self._geo.utm_to_local(mesh_utm)

        # Build KDTree
        search_coords = self._mesh_local if use_3d_search else self._mesh_local[:, :2]
        self._tree = cKDTree(search_coords)

        # Infer shapes
        self._num_tx = self._dataset.attrs.get("num_tx", 1)
        # Try to infer num_tx from path_gains if available
        pg_key = self._key_mapping.get("path_gains") or self._key_mapping.get(
            "path_gain"
        )
        if pg_key and pg_key in self._dataset:
            # path_gains: [RX, TX, ...]
            self._num_tx = self._dataset[pg_key].shape[1]

        # Load Metadata
        self._tx_positions = self._load_metadata("tx_positions")
        self._tx_orientations = self._load_metadata("tx_orientations")
        self._tx_antenna_gains = self._load_metadata("tx_antenna_gains")
        self._tx_names = self._load_metadata("tx_names")

        # Cache for best server mapping
        self._best_server_indices = None

    def _load_metadata(self, standard_key: str):
        """Helper to load metadata arrays if mapped."""
        key = self._key_mapping.get(standard_key)
        if key and key in self._dataset:
            return np.array(self._dataset[key])
        return None

    @property
    def tx_positions(self) -> Optional[np.ndarray]:
        """Returns the transmitter positions if available [NumTx, 3]."""
        return self._tx_positions

    @property
    def tx_orientations(self) -> Optional[np.ndarray]:
        """Returns the transmitter orientations if available [NumTx, 3] (Yaw, Pitch, Roll)."""
        return self._tx_orientations

    @property
    def tx_antenna_gains(self) -> Optional[np.ndarray]:
        """Returns the transmitter antenna gains if available [NumTx, 1]."""
        return self._tx_antenna_gains

    @property
    def tx_names(self) -> Optional[List[str]]:
        """Returns the transmitter names if available."""
        # Convert bytes to string if necessary
        if self._tx_names is not None:
            if self._tx_names.dtype.kind == "S" or self._tx_names.dtype.kind == "O":
                return [
                    n.decode("utf-8") if isinstance(n, bytes) else str(n)
                    for n in self._tx_names
                ]
            return self._tx_names
        return None

    @property
    def geo(self) -> CoordinateSystem:
        """Returns the coordinate system."""
        return self._geo

    def get_paths(self, ut_info: Union[np.ndarray, tf.Tensor, List[int]]) -> Paths:
        """
        Finds the nearest mesh points (if coordinates given) or uses indices directly,
        and returns an ExternalPaths object.

        Args:
            ut_info (Union[np.ndarray, tf.Tensor, List[int]]):
                Either:
                - User terminal coordinates [num_rx, 3] (float)
                - Mesh indices [num_rx] (int)
        """
        if isinstance(ut_info, tf.Tensor):
            ut_info = ut_info.numpy()
        else:
            ut_info = np.array(ut_info)

        # 1. Determine if input is indices or coordinates
        # If 1D and integer, assume indices
        if ut_info.ndim == 1 and np.issubdtype(ut_info.dtype, np.integer):
            indices = ut_info
        # If 2D [N, 3], assume coordinates
        elif ut_info.ndim == 2 and ut_info.shape[1] == 3:
            # Convert UTM to Local for KDTree query
            ut_coords_local = self._geo.utm_to_local(ut_info)

            search_coords = (
                ut_coords_local if self._use_3d_search else ut_coords_local[:, :2]
            )

            # Find nearest mesh point indices
            _, indices = self._tree.query(search_coords)
        else:
            raise ValueError(
                f"ut_info must be either [N, 3] coordinates or [N] indices. Got shape {ut_info.shape}"
            )

        # Instantiate ExternalPaths with the dataset and mapped indices
        return ExternalPaths(
            dataset=self._dataset,
            scene=self._scene,
            num_tx=self._num_tx,
            num_rx=len(indices),
            sample_index=indices,
            key_mapping=self._key_mapping,
        )

    def get_random_mesh_coordinates(self, num_uts: int) -> np.ndarray:
        """
        Randomly selects num_uts points from the available mesh points.
        """
        num_points = self._mesh_local.shape[0]
        indices = np.random.choice(num_points, size=num_uts, replace=False)
        return self._mesh_local[indices]

    def get_best_server_mapping(self) -> np.ndarray:
        """
        Pre-calculates the ID of the BS with the highest path gain for every mesh point.
        """
        if self._best_server_indices is not None:
            return self._best_server_indices

        if "path_gain" in self._store:
            # Shape: [Num_RX, Num_TX, Num_Paths]
            gains = np.array(self._store["path_gain"])
        elif "path_gains" in self._store:
            # Shape: [Num_RX, Num_TX, Num_Paths, 2, 2]
            pg = np.array(self._store["path_gains"])
            # Sum power over polarization and paths
            gains = np.sum(np.abs(pg) ** 2, axis=(-2, -1))
        else:
            raise KeyError("No gain data found for best server calculation")

        # Total gain per BS across all paths
        total_gains = np.sum(gains, axis=-1)  # [Num_RX, Num_TX]
        self._best_server_indices = np.argmax(total_gains, axis=1)  # [Num_RX]
        return self._best_server_indices

    def get_random_coordinates_by_best_server(
        self, bs_index: int, num_uts: int
    ) -> np.ndarray:
        """
        Randomly selects num_uts points from the coverage area of a specific BS.
        """
        mapping = self.get_best_server_mapping()
        candidate_indices = np.where(mapping == bs_index)[0]

        if len(candidate_indices) == 0:
            raise ValueError(
                f"No mesh points found where BS {bs_index} is the best server."
            )

        if len(candidate_indices) < num_uts:
            # If not enough points, just return all available (with warning-like behavior)
            indices = candidate_indices
        else:
            indices = np.random.choice(candidate_indices, size=num_uts, replace=False)

        return self._mesh_local[indices]


class SionnaLiveTracer(ExternalLoaderBase):
    """
    Wrapper for Sionna's real-time ray tracer.
    """

    def __init__(self, scene: Scene):
        """
        Args:
            scene (Scene): The Sionna scene object.
        """
        self._scene = scene

    def get_paths(self, ut_coordinates_local: Union[np.ndarray, tf.Tensor]) -> Paths:
        """
        Updates receiver positions and computes paths in real-time.
        """
        # Convert to numpy for position updates if needed
        if isinstance(ut_coordinates_local, tf.Tensor):
            ut_coords = ut_coordinates_local.numpy()
        else:
            ut_coords = ut_coordinates_local

        num_rx = ut_coords.shape[0]

        # Ensure correct number of receivers in the scene
        # This is a basic implementation; more complex logic might be needed
        # to handle antenna configurations per UT.
        rx_names = list(self._scene.receivers.keys())
        if len(rx_names) < num_rx:
            # We might need to add receivers, but for now we expect them to be pre-configured
            # or we update as many as we have.
            # Forwarding logic:
            pass

        for i in range(min(num_rx, len(rx_names))):
            self._scene.receivers[rx_names[i]].position = ut_coords[i]

        return self._scene.compute_paths()
