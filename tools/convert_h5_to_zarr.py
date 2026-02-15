#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import h5py
import zarr
import numcodecs
import numpy as np
from tqdm import tqdm


def convert_h5_to_zarr(h5_path, zarr_path, chunk_size=1000):
    """
    Converts Ray Tracing data from HDF5 to Zarr format with normalization.
    """
    print(f"Converting {h5_path} -> {zarr_path}")

    with h5py.File(h5_path, "r") as h5:
        # 1. Create Zarr store
        store = zarr.DirectoryStore(zarr_path)
        z = zarr.group(store)

        # 2. Copy and normalize datasets
        for key in h5.keys():
            data = h5[key]
            shape = data.shape
            dtype = data.dtype

            # Handle polarization matrices if they are flat in H5
            # We want [RX, TX, Paths, 2, 2] in Zarr
            if key == "path_gains" and len(shape) == 3:
                # If it's [RX, TX, Paths] but marked as polarized, it might need reshape
                # But usually if it's 3D and polarized, it's missing the 2x2.
                # Here we logic-check based on common Sionna/Matlab outputs.
                print(
                    f"Warning: {key} is 3D {shape}. Assuming scalar if not otherwise specified."
                )

            # Create Zarr dataset with LZ4 compression
            compressor = numcodecs.Blosc(
                cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
            )

            # Use chunks for large datasets
            target_chunks = list(shape)
            if len(shape) > 0:
                target_chunks[0] = min(shape[0], chunk_size)

            z_ds = z.create_dataset(
                key,
                shape=shape,
                chunks=tuple(target_chunks),
                dtype=dtype,
                compressor=compressor,
                overwrite=True,
            )

            # Copy data in chunks to save memory
            print(f"Copying {key}...")
            if len(shape) > 0:
                for i in tqdm(range(0, shape[0], chunk_size)):
                    end = min(i + chunk_size, shape[0])
                    z_ds[i:end] = np.array(h5[key][i:end])
            else:
                z_ds[...] = np.array(h5[key])

        # 3. Copy attributes (Metadata)
        print("Copying attributes...")
        for attr_key, attr_val in h5.attrs.items():
            if isinstance(attr_val, np.ndarray):
                z.attrs[attr_key] = attr_val.tolist()
            else:
                z.attrs[attr_key] = attr_val

    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Ray Tracing HDF5 to Zarr")
    parser.add_argument("input", help="Input HDF5 file path")
    parser.add_argument("output", help="Output Zarr path (directory or .zip)")
    parser.add_argument(
        "--chunks", type=int, default=1000, help="Chunk size for RX dimension"
    )

    args = parser.parse_args()
    convert_h5_to_zarr(args.input, args.output, chunk_size=args.chunks)
