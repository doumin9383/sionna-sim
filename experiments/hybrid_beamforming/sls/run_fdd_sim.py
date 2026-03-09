import os
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from experiments.hybrid_beamforming.sls.configs import SLSConfig, FDDConfig
from experiments.hybrid_beamforming.sls.simulator import SystemSimulator
from experiments.hybrid_beamforming.sls.components.fdd_data_generator import (
    FDDDataGenerator,
)
from experiments.hybrid_beamforming.sls.components.fdd_ml_model import build_model


def train_fdd_model(config, data_path):
    """Loads data and trains the selected ML model."""
    print(f"Loading training data from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Prepare Features and Targets
    # For MVP, we use Pattern A (UL SVD vectors) or Pattern B (Path info)
    # Target is always DL SVD 'v' vector

    X = []
    Y = []

    for sample in data:
        # UL SVD V: [B, BUT, F, TxP, Rank]
        ul_v = sample["ul_svd"]["v"]
        dl_v = sample["dl_svd"]["v"]
        paths = sample["ul_paths"]

        B_val, BUT_val, F_val, TxP_val, Rank_val = ul_v.shape
        num_ut_per_sector = config.num_ut_per_sector

        for b in range(B_val):
            for u in range(BUT_val):
                for f in range(F_val):
                    # Target: DL V (complex flatten)
                    # dl_v shape: [B, BUT, F, Ant, Rank]
                    target = np.concatenate(
                        [
                            np.real(dl_v[b, u, f]).flatten(),
                            np.imag(dl_v[b, u, f]).flatten(),
                        ]
                    )
                    Y.append(target)

                    # Features
                    if config.use_singular_vectors:
                        # Pattern A: UL V Vectors
                        feat = np.concatenate(
                            [
                                np.real(ul_v[b, u, f]).flatten(),
                                np.imag(ul_v[b, u, f]).flatten(),
                            ]
                        )
                        X.append(feat)
                    else:
                        # Pattern B: Path Features
                        # Indexing: paths['gain_abs'] is [B, num_bs, num_ut, C]
                        # We take the serving BS index
                        bs_idx = u // num_ut_per_sector
                        feat_list = []
                        # Note: Path features are usually same for all subbands in the same slot
                        if "gain_abs" in paths:
                            feat_list.append(paths["gain_abs"][b, bs_idx, u].flatten())
                        if "delay" in paths:
                            feat_list.append(paths["delay"][b, bs_idx, u].flatten())
                        if "aoa_azimuth" in paths:
                            feat_list.append(
                                paths["aoa_azimuth"][b, bs_idx, u].flatten()
                            )
                        if "aod_azimuth" in paths:
                            feat_list.append(
                                paths["aod_azimuth"][b, bs_idx, u].flatten()
                            )
                        X.append(np.concatenate(feat_list))

    X = np.array(X)
    Y = np.array(Y)

    print(f"Dataset shape: X={X.shape}, Y={Y.shape}")

    # Build and Train Model
    input_shape = X.shape[1:]
    output_dim = Y.shape[1]

    model = build_model(config.fdd_ml_model_type, input_shape, output_dim)

    if config.fdd_ml_model_type != "lightgbm":
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    else:
        model.train(X, Y)

    return model


def run_fdd_evaluation(config, ml_model=None):
    """Evaluates the FDD channel estimation against baselines."""
    # Modes to evaluate
    eval_modes = ["Ideal_DL", "Random", "UL_Reuse"]
    if ml_model is not None:
        eval_modes.append("ML_Predicted")

    results = {}

    for mode in eval_modes:
        print(f"--- Evaluating Mode: {mode} ---")
        config.precoding_strategy = mode

        # Initialize simulator for each mode.
        # This ensures fresh state and correct strategy application.
        sim = SystemSimulator(config, ml_model=ml_model)

        hist = sim.call(
            num_drops=config.num_ut_drops, tx_power_dbm=config.bs_max_power_dbm
        )
        results[mode] = hist["num_decoded_bits"].numpy()

    return results


def plot_fdd_results(results, output_dir):
    """Plots CDF of throughput for different modes."""
    plt.figure(figsize=(10, 6))
    for mode, thp in results.items():
        thp_flat = thp.flatten()
        sorted_thp = np.sort(thp_flat)
        y = np.arange(len(sorted_thp)) / float(len(sorted_thp))
        plt.plot(sorted_thp, y, label=mode)

    plt.xlabel("Throughput [Mbps]")
    plt.ylabel("CDF")
    plt.title("FDD UL-to-DL Channel Estimation Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fdd_comparison_cdf.png"))
    print(f"Result plot saved to {output_dir}")


def main():
    config = FDDConfig()

    # Step 1: Data Generation
    data_path = os.path.join(config.output_dir, "fdd_training_data.pkl")
    if not os.path.exists(data_path):
        print("Starting Data Generation Phase...")
        generator = FDDDataGenerator(config)
        # Collect data for training
        data = generator.collect_data(
            num_drops=config.num_ut_drops, num_slots_per_drop=1
        )
        generator.save_data(data, data_path)
    else:
        print(f"Using existing data at {data_path}")

    # Step 2: Training
    ml_model = train_fdd_model(config, data_path)

    # Step 3: Evaluation
    results = run_fdd_evaluation(config, ml_model)

    # Step 4: Plotting
    plot_fdd_results(results, config.output_dir)


if __name__ == "__main__":
    main()
