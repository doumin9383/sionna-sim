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


class MeshBasedLoader(ExternalLoaderBase):
    """
    Loads pre-computed ray tracing data from a mesh grid (Zarr/HDF5).

    Uses KDTree for nearest neighbor search between query points and mesh points.
    """

    def __init__(self, file_path: str, scene: Scene, use_3d_search: bool = False):
        """
        Args:
            file_path (str): Path to the Zarr store or HDF5 file.
            scene (Scene): The Sionna scene context.
            use_3d_search (bool): Whether to use (x, y, z) for nearest neighbor search.
                                  If False, only (x, y) is used.
        """
        self._file_path = file_path
        self._scene = scene
        self._use_3d_search = use_3d_search

        # Open store
        self._store = zarr.open(file_path, mode="r")

        # Initialize CoordinateSystem from metadata
        origin_utm = self._store.attrs.get("origin_utm", (0, 0, 0))
        self._geo = CoordinateSystem(origin_utm)

        # Load mesh coordinates (UTM) and convert to local
        if "mesh_coordinates" not in self._store:
            raise KeyError(f"Zarr store at {file_path} must contain 'mesh_coordinates'")

        mesh_utm = np.array(self._store["mesh_coordinates"])
        self._mesh_local = self._geo.utm_to_local(mesh_utm)

        # Build KDTree
        search_coords = self._mesh_local if use_3d_search else self._mesh_local[:, :2]
        self._tree = cKDTree(search_coords)

        # Infer shapes
        self._num_tx = self._store.attrs.get("num_tx", 1)
        if "path_gain" in self._store:
            self._num_tx = self._store["path_gain"].shape[1]
        elif "path_gains" in self._store:
            self._num_tx = self._store["path_gains"].shape[1]

        # Cache for best server mapping
        self._best_server_indices = None

    @property
    def geo(self) -> CoordinateSystem:
        """Returns the coordinate system."""
        return self._geo

    def get_paths(self, ut_coordinates_local: Union[np.ndarray, tf.Tensor]) -> Paths:
        """
        Finds the nearest mesh points and returns an ExternalPaths object.
        """
        if isinstance(ut_coordinates_local, tf.Tensor):
            ut_coords = ut_coordinates_local.numpy()
        else:
            ut_coords = ut_coordinates_local

        search_coords = ut_coords if self._use_3d_search else ut_coords[:, :2]

        # Find nearest mesh point indices
        _, indices = self._tree.query(search_coords)

        # Instantiate ExternalPaths with mapped indices
        # We assume sample_index in ExternalPaths handles lists/arrays of indices
        return ExternalPaths(
            zarr_path=self._file_path,
            scene=self._scene,
            num_tx=self._num_tx,
            num_rx=len(indices),
            sample_index=indices,
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
