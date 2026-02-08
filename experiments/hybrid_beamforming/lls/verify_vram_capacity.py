import os
import sys
import gc
import tensorflow as tf
import numpy as np
from experiments.hybrid_beamforming.lls.components.pusch_transmitter_wrapper import (
    HybridPUSCHTransmitter,
)
from sionna.phy.nr import PUSCHConfig


def verify_vram_capacity():
    print("Verifying VRAM Capacity for Narrowband Precoding...")

    # Configure GPU memory growth just in case
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs detected: {len(gpus)}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Running on CPU (VRAM test might be meaningless).")

    # Simulation parameters for heavy load
    carrier_frequency = 3.5e9
    subcarrier_spacing = 30e3
    num_tx_ant = 64  # High antenna count
    num_layers = 4  # High rank
    num_rb = 273  # Max RBs for 100MHz @ 30kHz SCS (approx) -> using 100 for typical heavy case
    # Actually, let's use a very heavy case to test limits
    num_rb = 106  # 40MHz

    pusch_config = PUSCHConfig(
        carrier_frequency=carrier_frequency,
        subcarrier_spacing=subcarrier_spacing,
        num_antenna_ports=num_layers,
        num_layers=num_layers,
        bandwidth=40e6,  # roughly 106 RBs
    )

    # Test batch sizes
    batch_sizes = [1, 10, 50, 100]

    max_safe_batch = 0

    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        try:
            # Instantiate Model
            transmitter = HybridPUSCHTransmitter(
                pusch_config=pusch_config,
                enable_transform_precoding=True,  # DFT-s-OFDM adds some compute
                num_tx_ant=num_tx_ant,
                precoding_granularity="Narrowband",  # The heaviest mode (RE-level)
            )

            # Run a forward pass
            print("  Running forward pass...")
            x = transmitter(bs)

            # Force execution
            _ = x.numpy()

            print(f"  Success! Output shape: {x.shape}")
            max_safe_batch = bs

            # Clean up
            del transmitter
            del x
            tf.keras.backend.clear_session()
            gc.collect()

        except Exception as e:
            print(f"  FAILED with Batch Size {bs}")
            print(f"  Error: {e}")
            break

    print(f"\nMaximum safe batch size estimated: {max_safe_batch}")


if __name__ == "__main__":
    verify_vram_capacity()
