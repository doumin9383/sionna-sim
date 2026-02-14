print("Start import test")
import sys
import os

# sys.path.append(os.getcwd())

try:
    print("Importing numpy...")
    import numpy as np

    print("Importing h5py...")
    import h5py

    print("Importing zarr...")
    import zarr

    print("Importing xarray...")
    import xarray

    print("Importing mitsuba...")
    import mitsuba as mi

    mi.set_variant("cuda_ad_mono_polarized")  # or llvm
    print("Importing drjit...")
    import drjit as dr

    print("Importing sionna...")
    import sionna
    from sionna.rt import Scene

    print("Importing wsim.rt.external...")
    from wsim.rt.external import HDF5Ingester, ExternalPaths

    print("All imports successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback

    traceback.print_exc()
