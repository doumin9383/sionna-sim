import os
import sys
import tensorflow as tf
import numpy as np

# --- Add project root to path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# sys.path.append(project_root)

from experiments.hybrid_beamforming.sls.simulator import HybridSystemSimulator
from experiments.hybrid_beamforming.sls.my_configs import HybridSLSConfig
from wsim.rt.configs import ResourceGridConfig
from sionna.phy.channel.tr38901 import PanelArray
from sionna.phy.ofdm import ResourceGrid


def run_probe():
    carrier_frequency = 3.5e9
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

    rg = ResourceGrid(
        num_ofdm_symbols=1,
        fft_size=24,
        subcarrier_spacing=30e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=6,
    )

    config = HybridSLSConfig(
        batch_size=1,
        num_rings=1,
        num_ut_per_sector=1,
        num_slots=1,
        carrier_frequency=carrier_frequency,
        resource_grid=rg,
        bs_array=bs_array,
        ut_array=ut_array,
        scenario="uma",
        direction="downlink",
        use_rbg_granularity=True,
        rbg_size_rb=4,
    )

    sim = HybridSystemSimulator(config=config)

    # Trace values in get_full_channel_info
    print("--- Probing HybridChannelInterface ---")
    h_port, s, u, v = sim.channel_interface.get_full_channel_info(1)

    def check(name, tensor):
        t = tensor.numpy() if isinstance(tensor, tf.Tensor) else tensor
        has_nan = np.any(np.isnan(t))
        print(f"{name}: shape={t.shape}, has_nan={has_nan}")
        if has_nan:
            print(f"  Sample values: {t.flatten()[:5]}")
        else:
            print(f"  Mean Amp: {np.mean(np.abs(t)):.4e}")

    check("h_port", h_port)
    check("s", s)
    check("u", u)
    check("v", v)

    print("\n--- Probing Interference Calculation ---")
    # Simulate parts of simulate_slot
    serving_bs_idx = tf.argmax(sim.stream_management.rx_tx_association, axis=1)
    serving_bs_idx_batched = tf.broadcast_to(serving_bs_idx, [1, sim.num_ut])

    u_serv = tf.gather(u, serving_bs_idx_batched, axis=2, batch_dims=2)
    check("u_serv", u_serv)

    # Interference path
    h_u = tf.einsum("buosrp,bujosrt->bujospt", tf.math.conj(u_serv), h_port)
    check("h_u", h_u)

    # SVD status
    print("\nChecking rank of channel matrices...")
    # Last two dims are rx_ports, tx_ports
    # s shape: [batch, ut, bs, ofdm, sc, min(ports)]
    print(f"s mean: {np.mean(s.numpy())}")
    if np.any(s.numpy() == 0):
        print("WARNING: Zero singular values found!")

    if np.any(np.isnan(s.numpy())):
        print("ERROR: SVD returned NaNs!")


if __name__ == "__main__":
    run_probe()
