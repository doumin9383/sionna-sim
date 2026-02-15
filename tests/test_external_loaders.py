import os
import numpy as np
import h5py
import zarr
import tensorflow as tf
from sionna.rt import Scene, Transmitter, Receiver, PlanarArray
from wsim.rt.external.loaders import MeshBasedLoader
from tools.convert_h5_to_zarr import convert_h5_to_zarr


def create_dummy_scene(num_tx=2, num_rx=10):
    scene = Scene()
    shared_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V",
    )
    scene.tx_array = shared_array
    scene.rx_array = shared_array
    for i in range(num_tx):
        tx = Transmitter(f"tx_{i}", position=[0, 0, 0])
        tx.antenna_array = shared_array
        scene.add(tx)
    for i in range(num_rx):
        rx = Receiver(f"rx_{i}", position=[0, 0, 0])
        rx.antenna_array = shared_array
        scene.add(rx)
    return scene


def test_hdf5_loader(h5_data, scene):
    h5_path, mesh, gains, delay = h5_data
    loader = MeshBasedLoader(h5_path, scene)

    # Query point exactly at mesh[0]
    ut_coords = loader.geo.origin_utm[np.newaxis, :] + loader._mesh_local[0:1]

    # Check KDTree search
    dist, indices = loader._tree.query(loader._mesh_local[0:1, :2])
    print(f"DEBUG H5: query UTM: {ut_coords}")
    print(f"DEBUG H5: KDTree indices: {indices}, dist: {dist}")

    paths = loader.get_paths(ut_coords)
    actual_tau = np.array(paths.tau)
    expected_tau = delay[0:1, np.newaxis, :, np.newaxis, :]

    print(f"DEBUG H5: actual_tau[0,0,0,0,:]: {actual_tau[0,0,0,0,:]}")
    print(f"DEBUG H5: expected_tau[0,0,0,0,:]: {expected_tau[0,0,0,0,:]}")

    np.testing.assert_allclose(actual_tau, expected_tau, atol=1e-5)


def test_zarr_loader_and_conversion(h5_data, scene, tmp_path):
    h5_path, mesh, gains, delay = h5_data
    zarr_path = str(tmp_path / "test.zarr")
    convert_h5_to_zarr(h5_path, zarr_path)

    loader = MeshBasedLoader(zarr_path, scene)

    # Query point exactly at mesh[1]
    ut_coords = loader.geo.origin_utm[np.newaxis, :] + loader._mesh_local[1:2]

    # Check KDTree search
    dist, indices = loader._tree.query(loader._mesh_local[1:2, :2])
    print(f"DEBUG ZARR: query UTM: {ut_coords}")
    print(f"DEBUG ZARR: KDTree indices: {indices}, dist: {dist}")

    paths = loader.get_paths(ut_coords)
    actual_tau = np.array(paths.tau)
    expected_tau = delay[1:2, np.newaxis, :, np.newaxis, :]

    print(f"DEBUG ZARR: actual_tau[0,0,0,0,:]: {actual_tau[0,0,0,0,:]}")
    print(
        f"DEBUG ZARR: expected_tau[1,0,0,0,:]: {delay[1,0,:]}"
    )  # This should match actual_tau[0]

    np.testing.assert_allclose(actual_tau, expected_tau, atol=1e-5)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    np.random.seed(42)  # Fixed seed

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        num_tx, num_rx_mesh, num_paths = 2, 10, 5
        h5_path = tmp_path / "test.h5"

        origin = np.array([100.0, 200.0, 10.0])
        mesh = np.random.rand(num_rx_mesh, 3).astype(np.float64)
        gains = (
            np.random.rand(num_rx_mesh, num_tx, num_paths, 2, 2)
            + 1j * np.random.rand(num_rx_mesh, num_tx, num_paths, 2, 2)
        ).astype(np.complex128)
        delay = np.random.rand(num_rx_mesh, num_tx, num_paths).astype(np.float64)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("mesh_coordinates", data=mesh)
            f.create_dataset("path_gains", data=gains)
            f.create_dataset("delay", data=delay)
            f.attrs["origin_utm"] = origin
            f.attrs["num_tx"] = num_tx

        scene = create_dummy_scene(num_tx=num_tx, num_rx=num_rx_mesh)
        h5_info = (str(h5_path), mesh, gains, delay)

        print("Running test_hdf5_loader...")
        test_hdf5_loader(h5_info, scene)
        print("test_hdf5_loader passed!")

        print("Running test_zarr_loader_and_conversion...")
        test_zarr_loader_and_conversion(h5_info, scene, tmp_path)
        print("test_zarr_loader_and_conversion passed!")

    print("All tests passed successfully!")
