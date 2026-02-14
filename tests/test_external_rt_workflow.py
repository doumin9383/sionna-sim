import os
import sys

# Add project root to sys.path
# sys.path.append(os.getcwd())
import shutil
import numpy as np
import h5py
import mitsuba as mi
import drjit as dr
import sionna
from sionna.rt import Scene

# Configure Mitsuba variant (must match Sionna's default)
try:
    mi.set_variant("cuda_ad_mono_polarized")
except Exception:
    mi.set_variant("llvm_ad_mono_polarized")

from wsim.rt.external import HDF5Ingester, ExternalPaths


def create_dummy_hdf5(path, num_samples=10, num_paths=5):
    """Creates a dummy HDF5 file with random ray tracing data."""
    with h5py.File(path, "w") as f:
        # Shapes: [num_samples, num_rx, num_tx, num_paths]
        # Simplified scenario: 1 RX, 1 TX
        shape = (num_samples, 1, 1, num_paths)

        f.create_dataset("path_gain", data=np.random.rand(*shape).astype(np.float32))
        f.create_dataset("delay", data=np.random.rand(*shape).astype(np.float32))
        f.create_dataset("zenith_at_tx", data=np.random.rand(*shape).astype(np.float32))
        f.create_dataset(
            "azimuth_at_tx", data=np.random.rand(*shape).astype(np.float32)
        )
        f.create_dataset("zenith_at_rx", data=np.random.rand(*shape).astype(np.float32))
        f.create_dataset(
            "azimuth_at_rx", data=np.random.rand(*shape).astype(np.float32)
        )


def test_workflow():
    work_dir = "test_external_rt_data"
    os.makedirs(work_dir, exist_ok=True)
    h5_path = os.path.join(work_dir, "dummy_rt.h5")
    zarr_path = os.path.join(work_dir, "ingested_rt.zarr")

    # 1. Create Dummy Data
    print("Creating dummy HDF5 data...")
    create_dummy_hdf5(h5_path)

    # 2. Ingest to Zarr
    print("Ingesting to Zarr...")
    ingester = HDF5Ingester(h5_path)
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    ingester.ingest_to_zarr(zarr_path, chunk_size=2)

    # 3. Load with ExternalPaths
    print("Loading with ExternalPaths...")

    # Needs a dummy scene for context (frequency/wavelength)
    scene = Scene()
    scene.frequency = 3.5e9  # 3.5 GHz

    # Add dummy TX and RX to satisfy Paths constructor
    # We need a configured scene for Paths to initialize its internal state correctly
    from sionna.rt import Transmitter, Receiver, PlanarArray

    # Create a simple antenna array
    array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    # Add one TX and one RX
    tx = Transmitter(name="tx", position=[0, 0, 0], orientation=[0, 0, 0])
    rx = Receiver(name="rx", position=[10, 0, 0], orientation=[0, 0, 0])
    scene.add(tx)
    scene.add(rx)
    scene.tx_array = array
    scene.rx_array = array

    try:
        paths = ExternalPaths(
            zarr_path=zarr_path,
            scene=scene,
            num_tx=1,
            num_rx=1,
            sample_index=0,  # Load the first scenario
        )
        print("ExternalPaths instantiated successfully.")

        # Verify data shape (roughly)
        # Expected shape after slice: [1, 1, num_paths]
        # (Assuming dim expansion didn't happen yet or handled by DrJit implicitly)
        print(f"Tau shape: {paths.tau.shape}")

    except Exception as e:
        print(f"Error loading paths: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    # shutil.rmtree(work_dir)


if __name__ == "__main__":
    test_workflow()
