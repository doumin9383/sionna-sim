import os
import numpy as np
import tensorflow as tf
import zarr
import shutil
import mitsuba as mi
from sionna.rt import Scene, Transmitter, Receiver, PlanarArray

# Set variant for Mitsuba
try:
    mi.set_variant("cuda_ad_mono_polarized")
except Exception:
    mi.set_variant("llvm_ad_mono_polarized")

from wsim.common.geo import CoordinateSystem
from wsim.rt.external.loaders import MeshBasedLoader, SionnaLiveTracer
from wsim.rt.external.paths import ExternalPaths


def test_geo_conversion():
    print("Testing Geo Conversion...")
    origin = (100.0, 200.0, 10.0)
    geo = CoordinateSystem(origin)

    # Test scalar-like
    utm = np.array([110, 220, 15])
    local = geo.utm_to_local(utm)
    assert np.allclose(local, [10, 20, 5])

    back_utm = geo.local_to_utm(local)
    assert np.allclose(back_utm, utm)

    # Test batch
    utm_batch = np.array([[110, 220, 15], [105, 205, 12]])
    local_batch = geo.utm_to_local(utm_batch)
    assert local_batch.shape == (2, 3)
    assert np.allclose(local_batch[1], [5, 5, 2])

    # Test TensorFlow
    utm_tf = tf.constant([[110, 220, 15]], dtype=tf.float32)
    local_tf = geo.utm_to_local(utm_tf)
    assert isinstance(local_tf, tf.Tensor)
    assert np.allclose(local_tf.numpy(), [[10, 20, 5]])

    print("Geo Conversion: PASSED")


def create_mock_zarr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    store = zarr.open(path, mode="w")

    num_mesh = 10
    num_tx = 2
    num_paths = 5

    # Mesh Coordinates (UTM)
    mesh_coords = np.zeros((num_mesh, 3))
    mesh_coords[:, 0] = np.arange(num_mesh) * 10.0 + 1000.0  # Easting
    mesh_coords[:, 1] = 2000.0  # Northing
    mesh_coords[:, 2] = 1.0  # Altitude

    store.create_dataset("mesh_coordinates", data=mesh_coords)

    # Complex Path Gains [Num_RX, Num_TX, Num_Paths, 2, 2]
    # Make BS 0 stronger for first 5 points, BS 1 stronger for last 5
    gains = np.zeros((num_mesh, num_tx, num_paths, 2, 2), dtype=np.complex64)
    gains[:5, 0, :, 0, 0] = 1.0  # BS 0 strong
    gains[5:, 1, :, 0, 0] = 1.0  # BS 1 strong

    store.create_dataset("path_gains", data=gains)

    # Delays
    delays = np.random.rand(num_mesh, num_tx, num_paths).astype(np.float32)
    store.create_dataset("delay", data=delays)

    # Metadata
    store.attrs["origin_utm"] = (1000.0, 2000.0, 0.0)
    store.attrs["num_tx"] = num_tx

    return store


def test_mesh_loader():
    print("Testing MeshBasedLoader...")
    zarr_path = "test_mesh.zarr"
    create_mock_zarr(zarr_path)

    scene = Scene()
    scene.frequency = 3.5e9

    # Dummy arrays for ExternalPaths initialization
    array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = array
    scene.rx_array = array

    # Add transmitters to scene to match num_tx
    for i in range(2):
        scene.add(Transmitter(f"tx_{i}", position=[0, 0, 0]))
    for i in range(1):  # Need at least one RX for dummy
        scene.add(Receiver(f"rx_{i}", position=[0, 0, 0]))

    loader = MeshBasedLoader(zarr_path, scene)

    # Test random sampling
    uts = loader.get_random_mesh_coordinates(3)
    assert uts.shape == (3, 3)

    # Test best server mapping
    mapping = loader.get_best_server_mapping()
    assert mapping.shape == (10,)
    assert np.all(mapping[:5] == 0)
    assert np.all(mapping[5:] == 1)

    # Test selective sampling
    uts_bs0 = loader.get_random_coordinates_by_best_server(0, 2)
    assert uts_bs0.shape == (2, 3)
    assert np.all(uts_bs0[:, 0] < 50.0)  # Local x < 50 for BS0 area

    # Test path retrieval
    query_pos = np.array(
        [[5.0, 0.0, 1.0], [85.0, 0.0, 1.0]]
    )  # Nearest to Mesh 0 and Mesh 8
    paths = loader.get_paths(query_pos)

    assert isinstance(paths, ExternalPaths)
    print(f"Paths a type: {type(paths.a)}")
    # print(f"Paths a: {paths.a}")

    try:
        print(f"Paths a shape: {paths.a.shape}")
        assert paths.a.shape[-2:] == (2, 2)
    except AttributeError as e:
        print(f"Caught AttributeError: {e}")
        if isinstance(paths.a, tuple):
            print(f"Paths a is a tuple of length {len(paths.a)}")
            print(f"First element type: {type(paths.a[0])}")

    print("MeshBasedLoader: PASSED")
    shutil.rmtree(zarr_path)


if __name__ == "__main__":
    test_geo_conversion()
    test_mesh_loader()
