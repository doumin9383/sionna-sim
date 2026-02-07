#
# Copyright (c) 2024 Sionna Sim (Contributor)
#

import os
import sys

# Add project root to sys.path to allow importing 'experiments' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from experiments.sls_end2end_hybrid_beam.components.pusch_model import (
    PUSCHCommunicationModel,
)


def run_papr_simulation(output_file="mpr_table.csv"):

    # Simulation Parameters
    batch_size = 100  # Adjust based on GPU memory
    num_batches = 10  # Total samples = 1000 slots

    # Scenarios to sweep
    scenarios = []

    # Define Modulation to MCS Index mapping (approximate for Table 1)
    # QPSK: 2, 16QAM: 11, 64QAM: 20, 256QAM: 28 (Sample indices)
    modulations = {"QPSK": 2, "16QAM": 11, "64QAM": 20, "256QAM": 28}

    # 1. CP-OFDM
    for mod_name, mcs_idx in modulations.items():
        for rank in [1, 2, 4]:
            scenarios.append(
                {
                    "waveform": "CP-OFDM",
                    "transform_precoding": False,
                    "modulation": mod_name,
                    "mcs_index": mcs_idx,
                    "rank": rank,
                }
            )

    # 2. DFT-s-OFDM (Transform Precoding)
    # Usually Rank 1 only for DFT-s-OFDM in typical usage (though MIMO is possible in Rel 16+)
    # We stick to Rank 1 for now as per instructions "DFT-s-OFDMは通常Rank 1"
    for mod_name, mcs_idx in modulations.items():
        scenarios.append(
            {
                "waveform": "DFT-s-OFDM",
                "transform_precoding": True,
                "modulation": mod_name,
                "mcs_index": mcs_idx,
                "rank": 1,
            }
        )

    results = []

    print(f"Starting PAPR Simulation with {len(scenarios)} scenarios...")

    for sc in tqdm(scenarios):
        # Instantiate Model
        # Note: num_tx_ant should be >= rank
        num_tx = 4
        if sc["rank"] > num_tx:
            num_tx = sc["rank"]  # Ensure enough antennas

        try:
            model = PUSCHCommunicationModel(
                carrier_frequency=3.5e9,
                subcarrier_spacing=30e3,
                num_tx_ant=num_tx,
                num_layers=sc["rank"],
                enable_transform_precoding=sc["transform_precoding"],
                mcs_index=sc["mcs_index"],
                papr_oversampling_factor=4,
            )

            papr_values = []

            for _ in range(num_batches):
                # Generate signal
                x = model.transmitter(batch_size)
                # x: [batch, tx, time]

                # Compute PAPR
                # Note: We compute for the oversampled signal inside compute_papr logic if integrated?
                # Using the integrated method in PUSCHCommunicationModel
                papr_db_batch = model.compute_papr(x)  # Returns [batch, tx]

                # Flatten and collect
                papr_values.extend(papr_db_batch.numpy().flatten())

            # Compute 99.9% CCDF (0.1% probability of exceeding)
            # Sort descending
            papr_sorted = np.sort(papr_values)
            # Index for 99.9%
            idx = int(0.999 * len(papr_sorted))
            papr_999 = papr_sorted[idx]

            # Record result
            res = sc.copy()
            res["papr_db_99.9"] = papr_999
            results.append(res)

        except Exception as e:
            print(f"Error in scenario {sc}: {e}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Simulation Complete. Results saved to {output_file}")
    print(df)


if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(
        os.path.dirname("lls_scripts/"), exist_ok=True
    )  # Create if running from root relative
    # Actually checking path
    run_papr_simulation()
