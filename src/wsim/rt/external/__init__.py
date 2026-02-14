#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
External Ray Tracing Data Support Module

This module provides functionality to ingest large-scale external ray tracing data
(e.g., from MATLAB/HDF5) into a format compatible with Sionna, utilizing
memory-efficient processing via xarray and Zarr.
"""

from .adapter import BaseAdapter, StandardAdapter
from .ingester import HDF5Ingester
from .paths import ExternalPaths

__all__ = [
    "BaseAdapter",
    "StandardAdapter",
    "HDF5Ingester",
    "ExternalPaths",
]
