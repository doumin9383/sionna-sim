import os
import sys
import tensorflow as tf

# --- Add project root to path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# Import Local Components
from experiments.sls_end2end_hybrid_beam.simulator import (
    HybridSystemSimulator,
)
from sionna.phy.channel.tr38901 import PanelArray
from sionna.phy.ofdm import ResourceGrid


def run_test():
    # 1. Configuration
    batch_size = 1
    carrier_frequency = 3.5e9
    subcarrier_spacing = 30e3
    fft_size = 24
    num_ofdm_symbols = 1
    num_ut_per_sector = 1
    num_rings = 1  # 1 site + 1 ring = 7 sites, 21 cells

    # 2. Arrays
    # BS: Single Panel 4x4, Cross-pol
    bs_array = PanelArray(
        num_rows=1,
        num_cols=1,
        num_rows_per_panel=4,
        num_cols_per_panel=4,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="38.901",
        carrier_frequency=carrier_frequency,
    )
    # UT: Single Panel 1x1, Cross-pol
    ut_array = PanelArray(
        num_rows=1,
        num_cols=1,
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
    )

    # 3. Resource Grid
    rg = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=subcarrier_spacing,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=6,
        pilot_pattern=None,  # No pilots for this MVP
    )  # No header

    # 4. Instantiate Simulator
    sim = HybridSystemSimulator(
        batch_size=batch_size,
        num_rings=num_rings,
        num_ut_per_sector=num_ut_per_sector,
        carrier_frequency=carrier_frequency,
        resource_grid=rg,
        scenario="uma",
        direction="downlink",
        ut_array=ut_array,
        bs_array=bs_array,
        bs_max_power_dbm=43.0,
        ut_max_power_dbm=23.0,
        coherence_time=10,  # slots
    )

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
    history = sim(num_slots, tx_power_dbm)

    print("Simulation completed.")
    print("History shape:", history.shape)

    # Save results to a file for inspection
    avg_tput = tf.reduce_mean(history, axis=[1, -1]).numpy()
    print("Average Metric per slot:", avg_tput)

    try:
        import csv

        with open("simulation_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Slot", "Average_Throughput_bps"])
            for i, val in enumerate(avg_tput):
                writer.writerow([i, val])
        print("Results saved to simulation_results.csv")
    except Exception as e:
        print(f"Failed to save results: {e}")

    print("Simulation completed.")
    print("History shape:", history.shape)
    avg_tput = tf.reduce_mean(history, axis=[1, -1]).numpy()  # Average over users
    print("Average Metric per slot:", avg_tput)


if __name__ == "__main__":
    run_test()
