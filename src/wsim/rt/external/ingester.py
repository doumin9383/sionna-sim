#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import h5py
import zarr
import numpy as np
import xarray as xr
from typing import Generator, Dict, Any, Tuple
from tqdm import tqdm
from .adapter import BaseAdapter, StandardAdapter


class HDF5Ingester:
    """
    Ingests large-scale HDF5 ray tracing data and converts it into a chunked Zarr store.
    """

    def __init__(self, h5_path: str, adapter: BaseAdapter = None):
        """
        Args:
            h5_path (str): Path to the source HDF5 file.
            adapter (BaseAdapter, optional): Adapter to map HDF5 keys. Defaults to StandardAdapter.
        """
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        self.adapter = adapter if adapter else StandardAdapter()
        self._inspect_metadata()

    def _inspect_metadata(self):
        """
        Lightweight inspection of the HDF5 file to validate schema and get shapes.
        """
        with h5py.File(self.h5_path, "r") as f:
            self.key_mapping = self.adapter.map_keys(f)

            # Assume all arrays have the same first dimension (num_samples or num_scenarios)
            # We check the shape of the first mapped key
            first_key = list(self.key_mapping.values())[0]
            self.total_samples = f[first_key].shape[0]

            print(f"Inspected {self.h5_path}: found {self.total_samples} samples.")
            print(f"Key mapping: {self.key_mapping}")

    def ingest_to_zarr(
        self, output_path: str, chunk_size: int = 100, overwrite: bool = False
    ):
        """
        Converts the HDF5 data to Zarr format with specified chunking.

        Args:
            output_path (str): Path to the output Zarr store (suffix .zarr or .zip).
            chunk_size (int): Number of samples per chunk in the primary dimension.
            overwrite (bool): Whether to overwrite existing output.
        """

        mode = "w" if overwrite else "w-"

        # Open HDF5 source
        with h5py.File(self.h5_path, "r") as source:

            # Prepare Zarr store
            store = zarr.open(output_path, mode=mode)

            # Create datasets in Zarr for each mapped key
            for std_key, h5_key in self.key_mapping.items():
                dataset = source[h5_key]
                shape = dataset.shape
                dtype = dataset.dtype

                # We chunk along the first dimension (batch/scenario)
                chunks = (chunk_size,) + shape[1:]

                # Initialize Zarr array
                z_array = store.create_dataset(
                    std_key,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
                )

                # Stream and write chunks
                print(f"Processing '{std_key}' from '{h5_key}'...")
                for i in tqdm(range(0, self.total_samples, chunk_size)):
                    end = min(i + chunk_size, self.total_samples)
                    # Read chunk from HDF5
                    data_chunk = dataset[i:end]
                    # Write chunk to Zarr
                    z_array[i:end] = data_chunk

            # Store metadata
            store.attrs["adapter_class"] = self.adapter.__class__.__name__
            store.attrs["original_file"] = self.h5_path

        print(f"Ingestion complete. Data stored at: {output_path}")

    @staticmethod
    def load_dataset(zarr_path: str) -> xr.Dataset:
        """
        Loads the Zarr store as an xarray Dataset (lazy loading).
        """
        ds = xr.open_zarr(zarr_path)
        return ds
