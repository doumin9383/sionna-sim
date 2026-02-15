#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import tensorflow as tf
from typing import Union, Tuple


class CoordinateSystem:
    """
    Coordinate conversion between global UTM and local simulation coordinates.

    UTM coordinates are typically (Easting, Northing, Altitude).
    Local coordinates are (x, y, z).

    The conversion is based on a fixed origin:
    Local = UTM - Origin
    UTM = Local + Origin
    """

    def __init__(
        self, origin_utm: Union[np.ndarray, tf.Tensor, Tuple[float, float, float]]
    ):
        """
        Initializes the coordinate system with a reference origin.

        Args:
            origin_utm (Union[np.ndarray, tf.Tensor, Tuple[float, float, float]]):
                The (Easting, Northing, Altitude) of the local origin in UTM.
        """
        if isinstance(origin_utm, (list, tuple)):
            self._origin = np.array(origin_utm, dtype=np.float64)
        elif isinstance(origin_utm, tf.Tensor):
            self._origin = origin_utm.numpy().astype(np.float64)
        else:
            self._origin = np.array(origin_utm, dtype=np.float64)

        if self._origin.shape != (3,):
            raise ValueError(
                f"Origin must have shape (3,), but got {self._origin.shape}"
            )

    @property
    def origin_utm(self) -> np.ndarray:
        """Returns the UTM origin as a numpy array."""
        return self._origin

    def utm_to_local(
        self, utm_coords: Union[np.ndarray, tf.Tensor]
    ) -> Union[np.ndarray, tf.Tensor]:
        """
        Converts global UTM coordinates to local simulation coordinates.

        Args:
            utm_coords (Union[np.ndarray, tf.Tensor]): UTM coordinates of shape [..., 3].

        Returns:
            Union[np.ndarray, tf.Tensor]: Local coordinates of shape [..., 3].
        """
        if isinstance(utm_coords, tf.Tensor):
            origin = tf.cast(self._origin, utm_coords.dtype)
            return utm_coords - origin
        else:
            return utm_coords - self._origin

    def local_to_utm(
        self, local_coords: Union[np.ndarray, tf.Tensor]
    ) -> Union[np.ndarray, tf.Tensor]:
        """
        Converts local simulation coordinates to global UTM coordinates.

        Args:
            local_coords (Union[np.ndarray, tf.Tensor]): Local coordinates of shape [..., 3].

        Returns:
            Union[np.ndarray, tf.Tensor]: UTM coordinates of shape [..., 3].
        """
        if isinstance(local_coords, tf.Tensor):
            origin = tf.cast(self._origin, local_coords.dtype)
            return local_coords + origin
        else:
            return local_coords + self._origin
