import tensorflow as tf
import numpy as np
import os
import pickle
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, PanelArray
from sionna.phy.ofdm import ResourceGrid
from wsim.common import weight_utils
from tqdm import tqdm


class FDDDataGenerator:
    """
    FDD UL/DL Channel Data Generator.
    Generates synchronized UL and DL channel data for machine learning.
    """

    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.num_ut = config.num_bs * config.num_ut_per_sector

        # Setup Resource Grid
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=1,
            fft_size=config.num_subcarriers,
            subcarrier_spacing=config.subcarrier_spacing,
            cyclic_prefix_length=config.cyclic_prefix_length,
        )

        # Setup Channel Models (UL and DL)
        self.ul_channel_model = self._setup_channel_model(config.ul_carrier_frequency)
        self.dl_channel_model = self._setup_channel_model(config.dl_carrier_frequency)

    def _setup_channel_model(self, carrier_frequency):
        common_params = {
            "carrier_frequency": carrier_frequency,
            "ut_array": self.config.ut_array,
            "bs_array": self.config.bs_array,
            "direction": self.config.direction,
            "enable_pathloss": True,
            "enable_shadow_fading": True,
            "precision": "single",
        }

        if self.config.scenario == "umi":
            return UMi(o2i_model="low", **common_params)
        elif self.config.scenario == "uma":
            return UMa(o2i_model="low", **common_params)
        elif self.config.scenario == "rma":
            return RMa(**common_params)
        else:
            raise ValueError(f"Unknown scenario: {self.config.scenario}")

    def generate_topology(self):
        """Generates random topology (UT locations etc.)"""
        from sionna.sys import gen_hexgrid_topology

        return gen_hexgrid_topology(
            batch_size=self.batch_size,
            num_rings=self.config.num_rings,
            num_ut_per_sector=self.config.num_ut_per_sector,
            min_bs_ut_dist=self.config.min_bs_ut_dist,
            max_bs_ut_dist=self.config.max_bs_ut_dist,
            scenario=self.config.scenario,
            precision=tf.float32,
        )

    def compute_svd_vectors(self, h, granularity="subband"):
        """
        Computes SVD singular vectors for a given channel.
        h: [batch, num_rx, num_tx, num_rx_ant, num_tx_ant, num_time, num_freq]
        """
        # Simplify dimensions for weight_utils
        # h_srv: [batch, num_ut, num_freq, num_rx_ant, num_tx_ant]
        # In this generator, we assume single-link or similar for simplicity,
        # or handle all UTs.

        # simulator.py process:
        # h_srv: [B, BUT, F, RxP, TxP]
        # Here h is [B, U, N, RxA, TxA, T, F]
        # We take Neighbor 0, Time 0
        h_srv = h[:, :, 0, :, :, 0, :]
        # Transpose to [B, U, F, RxA, TxA]
        h_srv = tf.transpose(h_srv, [0, 1, 4, 2, 3])

        # Compute SVD using weight_utils
        u, v, s = weight_utils.get_digital_precoders(
            h_srv,
            num_layers=self.config.num_layers,
            granularity=granularity,
            target_res=self.config.num_rb,  # N_target
            rbg_size_sc=(
                self.config.rbg_size_sc if hasattr(self.config, "rbg_size_sc") else 12
            ),
            weight_type="svd",
        )
        return u, v, s

    def extract_features_from_paths(self, paths):
        """
        Extracts Pattern B features: Gain, Delay, AoA, AoD.
        paths is a sionna.phy.channel.tr38901.Paths object.
        """
        features = {}
        # a: [batch, num_rx, num_tx, num_clusters, num_rays]
        if self.config.use_path_gain:
            features["gain_abs"] = tf.abs(paths.a)
            features["gain_phase"] = tf.math.angle(paths.a)

        if self.config.use_path_delay:
            features["delay"] = paths.tau

        if self.config.use_path_aoa:
            features["aoa_zenith"] = paths.theta_r
            features["aoa_azimuth"] = paths.phi_r

        if self.config.use_path_aod:
            features["aod_zenith"] = paths.theta_t
            features["aod_azimuth"] = paths.phi_t

        return features

    def collect_data(self, num_drops, num_slots_per_drop):
        """Main loop to collect UL/DL data pairs."""
        all_data = []

        sampling_frequency = 1 / self.resource_grid.ofdm_symbol_duration

        pbar = tqdm(total=num_drops * num_slots_per_drop, desc="Collecting Data")
        for drop in range(num_drops):
            # Generate Topology
            topo = self.generate_topology()
            ut_loc, bs_loc, ut_orient, bs_orient, ut_vel, in_state, los, _, _ = topo

            # Set same topology to both UL and DL models
            self.ul_channel_model.set_topology(
                ut_loc, bs_loc, ut_orient, bs_orient, ut_vel, in_state, los
            )
            self.dl_channel_model.set_topology(
                ut_loc, bs_loc, ut_orient, bs_orient, ut_vel, in_state, los
            )

            for slot in range(num_slots_per_drop):
                # Generate UL Channel
                ul_h, ul_paths = self.ul_channel_model(
                    num_samples=1, sampling_frequency=sampling_frequency
                )
                # Generate DL Channel
                dl_h, dl_paths = self.dl_channel_model(
                    num_samples=1, sampling_frequency=sampling_frequency
                )

                # Extract UL Features (Pattern B: Paths)
                ul_path_features = self.extract_features_from_paths(ul_paths)

                # Extract UL SVD (Pattern A)
                ul_u, ul_v, ul_s = self.compute_svd_vectors(ul_h)

                # Extract DL Targets (SVD vectors)
                dl_u, dl_v, dl_s = self.compute_svd_vectors(dl_h)

                # Store sample
                sample = {
                    "ul_paths": ul_path_features,
                    "ul_svd": {"u": ul_u.numpy(), "v": ul_v.numpy(), "s": ul_s.numpy()},
                    "dl_svd": {"u": dl_u.numpy(), "v": dl_v.numpy(), "s": dl_s.numpy()},
                    "drop_idx": drop,
                    "slot_idx": slot,
                }
                all_data.append(sample)
                pbar.update(1)
        pbar.close()

    def save_data(self, data, filename):
        """Saves collected data to a pickle file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")


if __name__ == "__main__":
    from experiments.hybrid_beamforming.sls.configs import SLSConfig

    config = SLSConfig()
    generator = FDDDataGenerator(config)
    data = generator.collect_data(num_drops=2, num_slots_per_drop=2)
    generator.save_data(data, os.path.join(config.output_dir, "fdd_training_data.pkl"))
