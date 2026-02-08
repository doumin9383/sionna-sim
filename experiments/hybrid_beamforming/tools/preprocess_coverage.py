import h5py
import zarr
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse


def preprocess_coverage(matlab_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Coverage Definition
    def_path = os.path.join(matlab_dir, "coverage_definition.mat")
    with h5py.File(def_path, "r") as f:
        # MeshPoints: [3 x num_mesh] -> [num_mesh x 3]
        mesh_points = np.array(f["MeshPoints"]).T
        mesh_ids = np.array(f["MeshIDs"]).flatten().astype(np.int32)

    num_mesh = len(mesh_ids)
    mesh_id_to_idx = {m_id: i for i, m_id in enumerate(mesh_ids)}

    # Save Mesh Points to Zarr
    z_root = zarr.open_group(os.path.join(output_dir, "coverage.zarr"), mode="w")
    z_root.create_dataset("mesh_points", data=mesh_points, chunks=(1000, 3))
    z_root.create_dataset("mesh_ids", data=mesh_ids, chunks=(1000,))

    # 2. Process Sectors (Power Map)
    power_files = sorted(glob.glob(os.path.join(matlab_dir, "sector_*_power.mat")))
    num_sectors = len(power_files)

    # Create a dense (but potentially sparse-backed) power matrix
    # [num_mesh, num_sectors]
    # For 900 sectors, this is manageable in memory if num_mesh is not in order of millions.
    # 1M * 900 * 4 bytes = 3.6 GB. manageable.
    power_matrix = np.full((num_mesh, num_sectors), -120.0, dtype=np.float32)

    print("Processing Power Maps...")
    for i, p_file in enumerate(tqdm(power_files)):
        with h5py.File(p_file, "r") as f:
            m_ids = np.array(f["MeshIDs"]).flatten().astype(np.int32)
            powers = np.array(f["Powers"]).flatten().astype(np.float32)

            for m_id, p in zip(m_ids, powers):
                if m_id in mesh_id_to_idx:
                    power_matrix[mesh_id_to_idx[m_id], i] = p

    z_root.create_dataset("power_map", data=power_matrix, chunks=(1000, num_sectors))

    # 3. Process Path Data
    # Path data is heterogeneous (num_paths varies).
    # We store them in a Zarr group per sector or a Ragged array.
    # Sector-wise group is easier to manage for 900 sectors.
    paths_group = z_root.create_group("paths")

    print("Processing Path Data...")
    path_files = sorted(glob.glob(os.path.join(matlab_dir, "sector_*_paths.mat")))
    for i, p_file in enumerate(tqdm(path_files)):
        sector_name = f"sector_{i:03d}"
        s_group = paths_group.create_group(sector_name)

        with h5py.File(p_file, "r") as f:
            # In our dummy/simplified real format:
            m_ids = np.array(f["MeshIDs"]).flatten().astype(np.int32)

            # Check if datasets exist (Gains, Delays, etc.)
            if "Gains" in f:
                # [num_paths x num_mesh] -> [num_mesh x num_paths]
                gains = np.array(f["Gains"]).T
                delays = np.array(f["Delays"]).T
                doa_az = np.array(f["DoA_Az"]).T
                doa_el = np.array(f["DoA_El"]).T
                dod_az = np.array(f["DoD_Az"]).T
                dod_el = np.array(f["DoD_El"]).T

                s_group.create_dataset("mesh_ids", data=m_ids)
                s_group.create_dataset("gains", data=gains.astype(np.complex64))
                s_group.create_dataset("delays", data=delays.astype(np.float32))
                s_group.create_dataset("doa_az", data=doa_az.astype(np.float32))
                s_group.create_dataset("doa_el", data=doa_el.astype(np.float32))
                s_group.create_dataset("dod_az", data=dod_az.astype(np.float32))
                s_group.create_dataset("dod_el", data=dod_el.astype(np.float32))

    print(f"Preprocessing complete. Metadata saved to {output_dir}/coverage.zarr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matlab_dir", default="data/matlab_dummy")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    preprocess_coverage(args.matlab_dir, args.output_dir)
