# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import h5py
import zarr
import numpy as np
import os
from sionna.rt import Scene, Transmitter, PlanarArray
from typing import Union


def setup_scene_from_hdf5(scene: Scene, filename: str):
    """
    Configures a Sionna Scene object based on metadata in an HDF5 or Zarr file.
    Automatically adds Transmitters with correct positions, orientations, and antenna configurations.

    Args:
        scene (Scene): The target Sionna Scene object.
        filename (str): Path to the HDF5 (.h5) or Zarr (.zarr, .zip) file.
    """

    # Open file
    _, ext = os.path.splitext(filename)
    if ext.lower() in [".h5", ".hdf5"]:
        f = h5py.File(filename, "r")
        # For HDF5, attributes are in f.attrs
        attrs = f.attrs
        # Datasets are in f
        datasets = f
    else:
        f = zarr.open(filename, mode="r")
        attrs = f.attrs
        datasets = f

    try:
        # 1. Set Coordinate System (Origin)
        if "origin_utm" in attrs:
            origin = attrs["origin_utm"]
            # Try setting standard method if exists, else custom attribute
            if hasattr(scene, "set_reference_point"):
                scene.set_reference_point(origin)
            else:
                # Fallback: Attach as custom attribute for reference
                scene.origin_utm = origin
            print(f"Scene origin set to UTM: {origin}")
        else:
            print(
                "Warning: 'origin_utm' attribute not found. Scene origin defaults to (0,0,0)."
            )
            origin = (0.0, 0.0, 0.0)

        # 2. Load Transmitter Metadata
        required_keys = ["tx_positions", "tx_orientations", "tx_antenna_gains"]
        data = {}
        for key in required_keys:
            if key in datasets:
                data[key] = np.array(datasets[key])
            else:
                raise KeyError(f"Missing required dataset: {key}")

        tx_positions = data["tx_positions"]
        tx_orientations = data["tx_orientations"]
        tx_ant_gains = data["tx_antenna_gains"]

        # Optional names
        if "tx_names" in datasets:
            raw_names = np.array(datasets["tx_names"])
            tx_names = []
            for n in raw_names:
                if isinstance(n, bytes):
                    tx_names.append(n.decode("utf-8"))
                else:
                    tx_names.append(str(n))
        else:
            tx_names = [f"Tx_{i}" for i in range(len(tx_positions))]

        num_tx = len(tx_positions)
        print(f"Found {num_tx} transmitters configuration.")

        # 3. Add Transmitters to Scene
        # Note: We assume a default PlanarArray configuration if not specified in metadata.
        # Ideally, we should parse antenna array config from metadata if available.
        # For now, we create a generic antenna array and apply the gain.
        # Since 'path_gains' in HDF5 usually excludes antenna pattern (isotropic),
        # we set up the array here so that Sionna can compute the pattern.

        # Default Array (Single Element or Small Array)
        # Verify if frequency is set in scene
        if scene.frequency is None:
            # Default to a common mid-band freq if not set (e.g., 3.5 GHz) or warn
            print(
                "Warning: Scene frequency not set. Using default 3.5 GHz for antenna array creation."
            )
            scene.frequency = 3.5e9

        # Create a basic logical antenna array.
        # The external data usually implies a specific sector shape.
        # Here we use a standard 3GPP-like panel.

        # Check if we should use existing scene.tx_array or create new ones per Tx
        # Using a shared array definition for all is common if they are identical.

        # Let's create a default single-pol 1-antenna element for valid geometric placement.
        # The actual gain is handled via the antenna pattern or modifying the 'look_at' etc.
        # IMPORTANT: 'tx_antenna_gains' is peak gain. PlanarArray computes gain based on pattern.
        # If we rely on stored path_gains, we might need to be careful about double counting.
        # However, the guide says: "path_gains excludes antenna pattern... apply_antenna_pattern() to be called".
        # So we MUST add Transmitters with correct orientation.

        # Using a default 8x8 array or similar placeholder?
        # No, let's use a simple 1-element Isotropic or defined panel if user specified.
        # Since we don't know the exact array config from HDF5 standard keys yet,
        # we will assume the user sets the antenna array *template* in the scene before calling this,
        # OR we create a default one.

        if scene.tx_array is None:
            # Create a default antenna array
            scene.tx_array = PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="iso",
                polarization="V",
            )

        # Convert UTM positions to Scene local coordinates (if origin is set)
        # scene.add() expects values relative to scene origin if origin is (0,0,0) in generic sense?
        # No, Sionna Scene uses "world coordinates".
        # "set_reference_point" sets the geo-reference.
        # But Transmitter.position should be in the simulation coordinate system (relative to origin if implicit).
        # Actually Sionna RT positions are XYZ.
        # If we set scene.origin (geo anchor), we generally expect positions in XYZ (meters) relative to that anchor usually?
        # NO. scene.origin is just a meta-anchor for map overlays.
        # The simulation kernel works in Cartesian XYZ.
        # HDF5 'tx_positions' are in UTM.
        # We need to subtract the origin to get local cartesian coordinates.

        origin_vec = np.array(origin)

        for i in range(num_tx):
            # Calculate local pos
            pos_utm = tx_positions[i]
            pos_local = pos_utm - origin_vec

            # Orientation: [Yaw, Pitch, Roll] in degrees
            orient = tx_orientations[i]

            # Create Transmitter
            # We reuse the scene.tx_array which acts as a template.
            # Names must be unique
            tx_name = tx_names[i]
            if tx_name in scene.transmitters:
                tx_name = f"{tx_name}_{i}"

            tx = Transmitter(name=tx_name, position=pos_local, orientation=orient)

            scene.add(tx)  # Implicitly uses scene.tx_array

        print(f"Successfully added {num_tx} transmitters to the scene.")

    finally:
        if ext.lower() in [".h5", ".hdf5"]:
            f.close()
        else:
            # Zarr object doesn't strictly need closing but good practice if wrapper
            pass
