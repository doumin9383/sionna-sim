import h5py
import numpy as np
import os


def create_dummy_hdf5(
    filename="dummy_rt_data.h5", num_mesh=100, num_tx=3, max_paths=10
):
    """
    Creates a dummy HDF5 file for testing Sionna-Sim RT ingestion.
    Follows the structure defined in wsim_rt_guide.md.
    """

    with h5py.File(filename, "w") as f:
        # 1. Path Data
        # path_gains: [N_Mesh, N_TX, MaxP, 2, 2] (Complex)
        # Random complex gains
        path_gains_real = np.random.randn(num_mesh, num_tx, max_paths, 2, 2).astype(
            np.float32
        )
        path_gains_imag = np.random.randn(num_mesh, num_tx, max_paths, 2, 2).astype(
            np.float32
        )
        path_gains = path_gains_real + 1j * path_gains_imag
        f.create_dataset("path_gains", data=path_gains)

        # delay: [N_Mesh, N_TX, MaxP] [s]
        # Random delays between 10ns and 1us
        delay = np.random.uniform(
            1e-8, 1e-6, size=(num_mesh, num_tx, max_paths)
        ).astype(np.float32)
        # Sort delays to look realistic (first path is shortest)
        delay = np.sort(delay, axis=-1)
        f.create_dataset("delay", data=delay)

        # angles: [N_Mesh, N_TX, MaxP] [rad]
        # zenith: 0 to pi, azimuth: -pi to pi
        for key in ["zenith_at_tx", "zenith_at_rx"]:
            data = np.random.uniform(
                0, np.pi, size=(num_mesh, num_tx, max_paths)
            ).astype(np.float32)
            f.create_dataset(key, data=data)

        for key in ["azimuth_at_tx", "azimuth_at_rx"]:
            data = np.random.uniform(
                -np.pi, np.pi, size=(num_mesh, num_tx, max_paths)
            ).astype(np.float32)
            f.create_dataset(key, data=data)

        # pathloss: [N_Mesh, N_TX] [dB]
        # Simple distance based + shadowing
        # Ideally calculated from path_gains sum, but here random
        pathloss = np.random.uniform(80, 120, size=(num_mesh, num_tx)).astype(
            np.float32
        )
        f.create_dataset("pathloss", data=pathloss)

        # 2. Metadata

        # tx_positions: [N_TX, 3] (UTM)
        # Assumed origin at 500000, 4000000
        origin = np.array([500000.0, 4000000.0, 30.0], dtype=np.float32)
        tx_pos_local = (
            np.random.randn(num_tx, 3).astype(np.float32) * 500
        )  # spread 500m
        tx_pos_local[:, 2] = 30.0  # Height 30m
        tx_positions = tx_pos_local + origin
        f.create_dataset("tx_positions", data=tx_positions)

        # tx_orientations: [N_TX, 3] [deg] [Yaw, Pitch, Roll]
        tx_orientations = np.zeros((num_tx, 3), dtype=np.float32)
        tx_orientations[:, 0] = np.random.uniform(0, 360, size=num_tx)  # Yaw
        f.create_dataset("tx_orientations", data=tx_orientations)

        # tx_antenna_gains: [N_TX, 1] [dBi]
        tx_ant_gains = np.full((num_tx, 1), 18.0, dtype=np.float32)
        f.create_dataset("tx_antenna_gains", data=tx_ant_gains)

        # tx_names: [N_TX] string (fixed length or vlen)
        # HDF5 strings are tricky. Using numpy S type (bytes)
        names = [f"BS_{i:03d}".encode("utf-8") for i in range(num_tx)]
        f.create_dataset("tx_names", data=np.array(names))

        # mesh_coordinates: [N_Mesh, 3]
        # Random points around origin
        mesh_pos_local = (
            np.random.randn(num_mesh, 3).astype(np.float32) * 200
        )  # spread 200m
        mesh_pos_local[:, 2] = 1.5  # UE height
        mesh_coordinates = mesh_pos_local + origin
        f.create_dataset("mesh_coordinates", data=mesh_coordinates)

        # Global Attributes
        f.attrs["origin_utm"] = origin
        f.attrs["mesh_step_m"] = 5.0
        f.attrs["num_tx"] = num_tx
        f.attrs["num_rx"] = num_mesh  # In this context mesh points are potential RXs

    print(f"Created dummy HDF5 file: {filename}")
    print(f"  Mesh Points: {num_mesh}")
    print(f"  Transmitters: {num_tx}")
    print(f"  Max Paths: {max_paths}")


if __name__ == "__main__":
    create_dummy_hdf5()
