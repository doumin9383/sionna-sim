import os
import sys
import tensorflow as tf
import mitsuba as mi

try:
    mi.set_variant("cuda_ad_rgb")
except Exception as e:
    print(f"Failed to set mitsuba variant: {e}")

sys.path.append(os.getcwd())

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))
from experiments.hybrid_beamforming.sls.configs import SLSConfig
from experiments.hybrid_beamforming.sls.simulator import SystemSimulator
from experiments.hybrid_beamforming.sls.external_loader import SLSExternalLoader
from wsim.rt.create_dummy_hdf5 import create_dummy_hdf5
from wsim.rt.external.ingester import HDF5Ingester


def verify_integration():
    print("=== Starting SLS Integration Verification ===")

    # Paths
    h5_path = "integration_test_data.h5"
    zarr_path = "integration_test_data.zarr"

    # 1. Create Dummy Data
    print("\n[Step 1] Creating Dummy HDF5 Data...")
    # Use fewer mesh points for speed, ensure appropriate num_tx
    create_dummy_hdf5(filename=h5_path, num_mesh=200, num_tx=3, max_paths=5)

    # 2. Ingest to Zarr
    print("\n[Step 2] Ingesting to Zarr...")
    ingester = HDF5Ingester(h5_path)
    ingester.ingest_to_zarr(zarr_path, overwrite=True)

    # 3. Configure SLS
    print("\n[Step 3] Configuring SLS...")
    config = SLSConfig()
    config.external_data_path = zarr_path
    config.batch_size = 1
    config.num_ut_drops = 1
    config.num_ut_per_sector = 2  # Total UT = 3 BS * 2 = 6 UTs. Mesh has 200, so safe.
    config.num_neighbors = 3  # Tx=3なので近傍探索数を調整
    config.carrier_frequency = 3.5e9
    config.output_dir = "integration_test_results"

    print(f"\n[DEBUG] Config Check:")
    print(f"  config.num_bs_ant: {config.num_bs_ant}")
    print(f"  config.bs_array.num_ant: {config.bs_array.num_ant}")

    # 4. Run Simulation
    print("\n[Step 4] Running Simulation with External Loader...")
    try:
        # Instantiate Simulator
        # Note: We pass the Class, not instance, as per our updated run_sim logic
        sim = SystemSimulator(config=config, external_loader=SLSExternalLoader)

        # Run 1 drop
        history = sim(config.num_ut_drops, config.bs_max_power_dbm)

        print("\n[SUCCESS] Simulation completed.")
        print("History keys:", history.keys())

        # Check if results contain expected metrics
        if "num_decoded_bits" in history:
            print("Decoded bits recorded.")
        else:
            print("[WARNING] num_decoded_bits missing.")

    except Exception as e:
        print(f"\n[FAIL] Simulation failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        # if os.path.exists(h5_path): os.remove(h5_path)
        # import shutil
        # if os.path.exists(zarr_path): shutil.rmtree(zarr_path)
        pass


if __name__ == "__main__":
    verify_integration()
