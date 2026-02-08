import numpy as np
import h5py
import os


def create_dummy_data(output_dir, num_sectors=21, num_mesh=1000):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Coverage Definition
    with h5py.File(os.path.join(output_dir, "coverage_definition.mat"), "w") as f:
        mesh_points = np.random.rand(num_mesh, 3) * 1000
        mesh_ids = np.arange(num_mesh).astype(np.int32)
        f.create_dataset("MeshPoints", data=mesh_points.T)  # MATLAB is column-major
        f.create_dataset("MeshIDs", data=mesh_ids.T)

    for s_id in range(num_sectors):
        # 2. Power Map
        # Pick 100 random meshes for this sector
        active_meshes = np.random.choice(mesh_ids, 100, replace=False)
        powers = np.random.uniform(-110, -50, 100)

        with h5py.File(
            os.path.join(output_dir, f"sector_{s_id:03d}_power.mat"), "w"
        ) as f:
            f.create_dataset("MeshIDs", data=active_meshes.T)
            f.create_dataset("Powers", data=powers.T)

        # 3. Path Data
        with h5py.File(
            os.path.join(output_dir, f"sector_{s_id:03d}_paths.mat"), "w"
        ) as f:
            f.create_dataset("MeshIDs", data=active_meshes.T)

            # For each mesh, L paths.
            L = 10
            gains = (np.random.randn(100, L) + 1j * np.random.randn(100, L)) * 1e-6
            delays = np.random.uniform(0, 1e-6, (100, L))
            doa_az = np.random.uniform(0, 2 * np.pi, (100, L))
            doa_el = np.random.uniform(0, np.pi, (100, L))
            dod_az = np.random.uniform(0, 2 * np.pi, (100, L))
            dod_el = np.random.uniform(0, np.pi, (100, L))

            f.create_dataset("Gains", data=gains.T)
            f.create_dataset("Delays", data=delays.T)
            f.create_dataset("DoA_Az", data=doa_az.T)
            f.create_dataset("DoA_El", data=doa_el.T)
            f.create_dataset("DoD_Az", data=dod_az.T)
            f.create_dataset("DoD_El", data=dod_el.T)


if __name__ == "__main__":
    create_dummy_data("data/matlab_dummy")
    print("Dummy MATLAB data created in data/matlab_dummy (21 sectors, with angles)")
