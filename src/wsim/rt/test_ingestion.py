from wsim.rt.external.ingester import HDF5Ingester


def test_ingestion():
    input_file = "dummy_rt_data.h5"
    output_file = "dummy_rt_data.zarr"

    print(f"Ingesting {input_file} to {output_file}...")
    ingester = HDF5Ingester(input_file)
    ingester.ingest_to_zarr(output_file, overwrite=True)

    import zarr

    store = zarr.open(output_file, mode="r")
    print("Zarr store structure:")
    print(store.tree())

    # Check key existence
    expected_keys = [
        "path_gains",
        "delay",
        "zenith_at_tx",
        "azimuth_at_rx",
        "tx_positions",
    ]
    for k in expected_keys:
        if k in store:
            print(f"[OK] Key found: {k}, Shape: {store[k].shape}")
        else:
            print(f"[FAIL] Key missing: {k}")


if __name__ == "__main__":
    test_ingestion()
