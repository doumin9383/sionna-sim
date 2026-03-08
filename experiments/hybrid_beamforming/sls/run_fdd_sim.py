import os
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from experiments.hybrid_beamforming.sls.configs import SLSConfig
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
        # UL Features
        if config.use_singular_vectors:
            # Flatten UL V vector: [BUT, F, TxP, Rank]
            ul_v = sample["ul_svd"]["v"]
            X.append(ul_v.flatten())
        else:
            # Use Path Features (Pattern B)
            # Simplification: flattening all path info
            paths = sample["ul_paths"]
            feat_list = []
            if "gain_abs" in paths:
                feat_list.append(paths["gain_abs"].flatten())
            if "delay" in paths:
                feat_list.append(paths["delay"].flatten())
            if "aoa_azimuth" in paths:
                feat_list.append(paths["aoa_azimuth"].flatten())
            if "aod_azimuth" in paths:
                feat_list.append(paths["aod_azimuth"].flatten())
            X.append(np.concatenate(feat_list))

        # Target: DL V vector (flattened)
        dl_v = sample["dl_svd"]["v"]
        # Convert complex to real/imag for regression
        # dl_v: [BUT, F, TxP, Rank]
        target = np.concatenate([np.real(dl_v).flatten(), np.imag(dl_v).flatten()])
        Y.append(target)

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
        results[mode] = hist["throughput_per_user"]

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
    config = SLSConfig()
    config.num_ut_drops = 5  # Sufficient for basic CDF
    config.num_slots = 5
    config.precoding_granularity = "Wideband"

    # Step 1: Data Generation
    data_path = os.path.join(config.output_dir, "fdd_training_data.pkl")
    if not os.path.exists(data_path):
        print("Starting Data Generation Phase...")
        generator = FDDDataGenerator(config)
        # Collect enough data for training
        data = generator.collect_data(num_drops=10, num_slots_per_drop=10)
        generator.save_data(data, data_path)
    else:
        print(f"Using existing data at {data_path}")

    # Step 2: Training
    ml_model = train_fdd_model(config, data_path)

    # Step 3: Evaluation
    # Note: SystemSimulator now accepts ml_model in __init__
    results = run_fdd_evaluation(config, ml_model)

    # Step 4: Plotting
    plot_fdd_results(results, config.output_dir)


if __name__ == "__main__":
    main()
