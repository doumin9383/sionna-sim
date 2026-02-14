import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import zarr
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
# sys.path.append(project_root)

from sionna.sys import gen_hexgrid_topology
from sionna.phy.channel.tr38901 import UMi, UMa, RMa
from experiments.hybrid_beamforming.sls.my_configs import HybridSLSConfig


def generate_dataset(output_path, num_drops=10):
    print(f"Generating dataset with {num_drops} drops to {output_path}...")

    # 1. Configuration
    config = HybridSLSConfig()
    config.batch_size = 1  # Force batch size 1 for drop-by-drop generation

    # 2. Channel Model with return_rays=True
    # We need to instantiate the model similar to HybridSystemSimulator
    if config.scenario == "umi":
        model = UMi(
            carrier_frequency=config.carrier_frequency,
            o2i_model="low",
            ut_array=config.ut_array,
            bs_array=config.bs_array,
            direction=config.direction,
            enable_pathloss=True,
            enable_shadow_fading=True,
        )
    elif config.scenario == "uma":
        model = UMa(
            carrier_frequency=config.carrier_frequency,
            o2i_model="low",
            ut_array=config.ut_array,
            bs_array=config.bs_array,
            direction=config.direction,
            enable_pathloss=True,
            enable_shadow_fading=True,
        )
    # Add RMa if needed

    # Enable Ray return
    model.return_rays = True

    # 3. Zarr Storage Initialization
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)

    # 4. Loop over drops
    # Force CPU for generation to avoid OOM
    with tf.device("/CPU:0"):
        for i in tqdm(range(num_drops), desc="Generating Drops"):
            # a. Generate Topology
            topology = gen_hexgrid_topology(
                batch_size=config.batch_size,
                num_rings=config.num_rings,
                num_ut_per_sector=config.num_ut_per_sector,
                min_bs_ut_dist=config.min_bs_ut_dist,
                max_bs_ut_dist=config.max_bs_ut_dist,
                scenario=config.scenario,
                los=None,  # Mixed LoS/NLoS
                return_grid=True,  # Match simulator.py behavior
                precision=model.precision,
            )
            # topology is a tuple of 9 items when return_grid=True
            (ut_loc, bs_loc, ut_orient, bs_orient, ut_vel, in_state, los, _, _) = (
                topology
            )

            # If gen_hexgrid_topology returns None for los, we need to retrieve it
            # from the model after the first set_topology call.
            # We will set the topology for the full set of UTs once to get the LoS status.
            # This is a temporary set_topology call just to get the LoS status if it was None.
            if los is None:
                model.set_topology(
                    ut_loc, bs_loc, ut_orient, bs_orient, ut_vel, in_state, None
                )
                los = model._scenario.los  # Retrieve the generated LoS from the model

            # Save Topology per drop
            # Group: drop_{i}/topology
            grp_drop = root.create_group(f"drop_{i}")
            grp_topo = grp_drop.create_group("topology")
            grp_topo.create_dataset("ut_loc", data=ut_loc.numpy())
            grp_topo.create_dataset("bs_loc", data=bs_loc.numpy())
            grp_topo.create_dataset("ut_orient", data=ut_orient.numpy())
            grp_topo.create_dataset("bs_orient", data=bs_orient.numpy())
            grp_topo.create_dataset("ut_vel", data=ut_vel.numpy())
            grp_topo.create_dataset("in_state", data=in_state.numpy())
            grp_topo.create_dataset("los", data=los.numpy())

            # Calculating Serving Cell (All UTs)
            diff = tf.expand_dims(ut_loc, 2) - tf.expand_dims(bs_loc, 1)
            dists = tf.norm(diff, axis=-1)
            serving_cell_ids = tf.argmin(dists, axis=2, output_type=tf.int32).numpy()
            grp_topo.create_dataset("serving_cell_id", data=serving_cell_ids)

            # b. Loop over UTs in batches (to avoid OOM during Ray Generation)
            ut_batch_size_gen = 4  # Process 4 UTs at a time
            num_ut_total = ut_loc.shape[1]

            # Temporary lists to collect rays and LoS for this drop
            drop_rays = {
                "delays": [],
                "powers": [],
                "aoa": [],
                "aod": [],
                "zoa": [],
                "zod": [],
                "xpr": [],
            }
            drop_lsps = {
                "pathloss": [],
                "shadow_fading": [],
                "k_factor": [],
            }
            drop_los = []

            for u_start in range(0, num_ut_total, ut_batch_size_gen):
                u_end = min(u_start + ut_batch_size_gen, num_ut_total)

                # Slice Topology
                sub_ut_loc = ut_loc[:, u_start:u_end, :]
                sub_ut_orient = ut_orient[:, u_start:u_end, :]
                sub_ut_vel = ut_vel[:, u_start:u_end, :]
                sub_in_state = in_state[:, u_start:u_end]
                # We do NOT pass sub_los. We let the model accept None and generate it.

                # Set Topology for this subset of UTs (against ALL BSs)
                model.set_topology(
                    sub_ut_loc,
                    bs_loc,
                    sub_ut_orient,
                    bs_orient,
                    sub_ut_vel,
                    sub_in_state,
                    los=None,  # Let model generate LoS
                )

                # Retrieve the generated LoS for consistency
                current_los = model._scenario.los  # [Batch, NumBS, SubUT]
                drop_los.append(current_los.numpy())

                # Sample Pathloss and Retrieve LSPs
                # We need to manually sample Pathloss for saving (model calls it internally but doesn't store it)
                # Since PL has random O2I component, we must use the one we save.
                current_pl = (
                    model._lsp_sampler.sample_pathloss()
                )  # [Batch, NumBS, SubUT]

                # Retrieve generated SF and K-factor from model._lsp
                # model._lsp is updated by set_topology
                current_sf = model._lsp.sf  # [Batch, NumBS, SubUT]
                current_k = model._lsp.k_factor  # [Batch, NumBS, SubUT]

                drop_lsps["pathloss"].append(current_pl.numpy())
                drop_lsps["shadow_fading"].append(current_sf.numpy())
                drop_lsps["k_factor"].append(current_k.numpy())

                # Run Model
                # num_time_samples=1
                ret = model(num_time_samples=1, sampling_frequency=30e3)
                _, _, rays = ret  # h, delays, rays

                # Append to list
                # rays attributes are [batch, num_rx, num_tx, num_paths]
                # We want to stack along dimension 1 (Rx/UT)
                drop_rays["delays"].append(rays.delays.numpy())
                drop_rays["powers"].append(rays.powers.numpy())
                drop_rays["aoa"].append(rays.aoa.numpy())
                drop_rays["aod"].append(rays.aod.numpy())
                drop_rays["zoa"].append(rays.zoa.numpy())
                drop_rays["zod"].append(rays.zod.numpy())
                drop_rays["xpr"].append(rays.xpr.numpy())

            # Concatenate collected rays for the drop
            # Axis 1 is UT/Rx
            grp_rays = grp_drop.create_group("rays")
            for key in drop_rays:
                # Concatenate along axis 1 (UT dimension)
                # Each element in list is [1, sub_ut, bs, paths]
                if drop_rays[key]:
                    data = np.concatenate(drop_rays[key], axis=1)
                    grp_rays.create_dataset(key, data=data)

            # Save LSPs
            # These are [Batch, Terminals(BS), SubUT] -> Transpose/Reshape needed?
            # model._lsp shape is [Batch, NumBS, NumUT] (or Rx/Tx depending on direction)
            # UMi Downlink: Rx=UT, Tx=BS. LSPs usually [Batch, Tx, Rx] -> [Batch, NumBS, NumUT]
            # concat along axis 2 (UT)
            for key in drop_lsps:
                if drop_lsps[key]:
                    data = np.concatenate(drop_lsps[key], axis=2)
                    grp_rays.create_dataset(key, data=data)

            # Concatenate and save consistent LoS
            # current_los is [Batch, NumBS, SubUT]. We concat along axis 2 (UT)
            if drop_los:
                final_los = np.concatenate(drop_los, axis=2)
                # Overwrite/Create "los" dataset in topology group
                if "los" in grp_topo:
                    del grp_topo["los"]
                grp_topo.create_dataset("los", data=final_los)

    print("Dataset generation complete.")


if __name__ == "__main__":
    generate_dataset("data/processed/calibration_sls.zarr", num_drops=2)
