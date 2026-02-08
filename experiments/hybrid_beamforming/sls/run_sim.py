import os
import sys
import tensorflow as tf

# --- Add project root to path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

# Import Local Components
from experiments.hybrid_beamforming.sls.simulator import (
    HybridSystemSimulator,
)
from experiments.hybrid_beamforming.sls.my_configs import HybridSLSConfig
from libs.my_configs import ResourceGridConfig
from sionna.phy.channel.tr38901 import PanelArray
from sionna.phy.ofdm import ResourceGrid
import csv


def run_test():
    # Master Config
    config = HybridSLSConfig(
        batch_size=1,
        num_rings=1,
        num_ut_per_sector=1,
        num_slots=1,
        scenario="uma",
        direction="downlink",
        coherence_time=10,
        num_neighbors=4,
    )

    # Resource Grid Config (From config)
    rg_config = config.resource_grid
    rg = ResourceGrid(
        num_ofdm_symbols=rg_config.num_ofdm_symbols,
        fft_size=rg_config.fft_size,
        subcarrier_spacing=rg_config.subcarrier_spacing,
        num_tx=rg_config.num_tx,
        # In Sionna SLS, usually num_tx in ResourceGrid matches the number of streams or transmitters depending on usage.
        # But here we are setting up the system.
        # Let's keep num_tx=1 for the grid definition unless specific needs arise.
        num_streams_per_tx=rg_config.num_streams_per_tx,
        cyclic_prefix_length=rg_config.cyclic_prefix_length,
        pilot_pattern="kronecker",  # Defaulting to kronecker if not in config or explicitly needed
        pilot_ofdm_symbol_indices=rg_config.pilot_ofdm_symbol_indices,
    )
    config.resource_grid = rg

    # BS Array from config
    bs_array = PanelArray(
        num_rows=config.bs_num_rows_panel,
        num_cols=config.bs_num_cols_panel,
        num_rows_per_panel=config.bs_num_rows_per_panel,
        num_cols_per_panel=config.bs_num_cols_per_panel,
        polarization=config.bs_polarization,
        polarization_type="cross",
        antenna_pattern="38.901",
        carrier_frequency=config.carrier_frequency,
    )
    # UT Array from config
    ut_array = PanelArray(
        num_rows=config.ut_num_rows_panel,
        num_cols=config.ut_num_cols_panel,
        num_rows_per_panel=config.ut_num_rows_per_panel,
        num_cols_per_panel=config.ut_num_cols_per_panel,
        polarization=config.ut_polarization,
        polarization_type="cross",
        antenna_pattern="omni",
        carrier_frequency=config.carrier_frequency,
    )

    config.bs_array = bs_array
    config.ut_array = ut_array

    # 4. Instantiate Simulator
    sim = HybridSystemSimulator(config=config)

    # 5. Run Simulation
    # 5. Run Simulation
    print("Starting simulation (Small Setup)...")
    num_slots = 1  # Reduced for quick verification
    tx_power_dbm = 43.0

    # Enable XLA for potential speedup if available, but for debugging eager might be safer
    # tf.config.optimizer.set_jit(True)

    # Run
    # Note: simulator.py loops internally.
    # To see progress, we might need to modify simulator.py or just trust it returns quickly.
    # With num_slots=1, it should be fast.
    # Run
    # Returns a dictionary of Tensors
    history = sim(num_slots, tx_power_dbm)

    print("Simulation completed.")
    print("History keys:", history.keys())

    # Save results to a pickle file for comprehensive analysis
    import pickle

    os.makedirs(config.output_dir, exist_ok=True)
    history_path = os.path.join(config.output_dir, "history.pkl")

    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Full history saved to {history_path}")

    # Calculate Average Throughput for quick check
    # num_decoded_bits: [slots, batch, bs, ut_per_sector]
    if "num_decoded_bits" in history:
        bits = history["num_decoded_bits"]
        # Sum over users (axis 2, 3) and batch (axis 1) -> [slots]
        # Or mean over users? Usually total network throughput
        total_bits_per_slot = tf.reduce_sum(bits, axis=[1, 2, 3])
        avg_tput_mbps = (
            tf.reduce_mean(total_bits_per_slot) / 1e6 * (1.0 / sim.slot_duration)
        )  # Approximate rate?
        # Actually bits is per slot. Rate = bits / duration.
        # But we stored "throughput_per_user" (bps) into num_decoded_bits in simulator.py
        # So it is already rate (bps).
        total_throughput_bps = tf.reduce_sum(bits, axis=[1, 2, 3])
        avg_throughput_mbps = tf.reduce_mean(total_throughput_bps) / 1e6
        print(f"Average Network Throughput: {avg_throughput_mbps:.2f} Mbps")

    # Simple CSV export for legacy plotting compatibility
    try:
        csv_path = os.path.join(config.output_dir, "simulation_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Slot", "Average_Throughput_bps"])
            # Writing total network throughput per slot
            tput_vals = total_throughput_bps.numpy()
            for i, val in enumerate(tput_vals):
                writer.writerow([i, val])
        print(f"Summary CSV saved to {csv_path}")
    except Exception as e:
        print(f"Failed to save summary CSV: {e}")


if __name__ == "__main__":
    run_test()
