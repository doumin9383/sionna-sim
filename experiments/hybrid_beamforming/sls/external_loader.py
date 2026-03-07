# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import numpy as np
import logging
import drjit as dr
import mitsuba as mi
from sionna.rt import Scene

# ...

# ...

# ...

from wsim.rt.external.loaders import MeshBasedLoader
from wsim.rt.scene_setup import setup_scene_from_hdf5
from experiments.hybrid_beamforming.sls.configs import SLSConfig


class SLSExternalLoader:
    """
    Adapter class to load external Ray Tracing data (HDF5/Zarr) into the SLS SystemSimulator.

    It wraps the MeshBasedLoader to provide:
    1. Topology information (BS/UT locations) compatible with SLS needs.
    2. Path data via ExternalPaths for the HybridChannelInterface.
    3. Management of random UT drops within the mesh.
    """

    def __init__(self, config: SLSConfig):
        self.config = config
        self.logger = logging.getLogger("SLSExternalLoader")

        path = config.external_data_path
        if not path:
            raise ValueError(
                "SLSConfig.external_data_path must be set for SLSExternalLoader."
            )

        # 1. Setup Scene (dummy scene to hold transmitters)
        self.scene = Scene()
        # Ensure frequency is set
        self.scene.frequency = config.carrier_frequency

        # Load Transmitters from metadata
        setup_scene_from_hdf5(self.scene, path)

        # 2. Initialize MeshBasedLoader
        # We assume 2D search is sufficient for typical terrestrial SLS
        self.loader = MeshBasedLoader(
            file_path=path, scene=self.scene, use_3d_search=False
        )

        # Cache BS locations from scene (simulating "Global" generic coordinates)
        # Note: setup_scene_from_hdf5 sets positions relative to implicit origin if set_reference_point fails,
        # or relative to origin if it succeeds.
        # But MeshLoader deals with UTM usually.
        # Let's rely on loader's properties if implementation allows, or extract from scene.

        # The loader.tx_positions are loaded from metadata (UTM).
        self.bs_loc_utm = self.loader.tx_positions  # [N_BS, 3]
        self.bs_orient = self.loader.tx_orientations  # [N_BS, 3]

        if self.bs_loc_utm is None:
            # Fallback if loader didn't load metadata?
            # setup_scene_from_hdf5 worked, so metadata exists.
            # but loader._tx_positions might be None if key map failed?
            # Let's trust loader.
            raise ValueError("Could not load TX positions from external file.")

        self.num_bs = self.bs_loc_utm.shape[0]

        # UT state buffer
        self.current_ut_idx = None  # Indices in the mesh
        self.current_ut_loc_utm = None

        # Initialize a dummy channel model (UMa) to borrow _cir_sampler and direction
        # HybridChannelInterface uses _cir_sampler to convert Rays to CIR.
        # We need to provide this method.
        # Check if we can import from standard path
        from sionna.phy.channel.tr38901 import UMa, PanelArray

        # Initialize a dummy channel model (UMa) to borrow _cir_sampler and direction
        # HybridChannelInterface uses _cir_sampler to convert Rays to CIR.
        # We need to provide this method.
        # Check if we can import from standard path
        from sionna.phy.channel.tr38901 import UMa, PanelArray

        # Use arrays from config to ensure dimensions match the internal simulator expectations
        self.base_model = UMa(
            carrier_frequency=config.carrier_frequency,
            o2i_model="low",
            ut_array=config.ut_array,
            bs_array=config.bs_array,
            direction="uplink",
            enable_pathloss=False,
            enable_shadow_fading=False,
        )

    @property
    def _cir_sampler(self):
        return self.base_model._cir_sampler

    @property
    def direction(self):
        return self.base_model.direction

    def load_drop(self, drop_idx: int):
        """
        Prepares a new random drop of UEs.
        Selection logic:
        - If config.num_ut is set implies total UTs?
        - usually SLS config specifies num_ut_per_sector.
        - Total UTs = num_bs * num_ut_per_sector (if full loading)

        Here we simply select N random points from the mesh to serve as UEs.
        """
        # Determine number of UEs
        # Config has num_ut_per_sector.
        # Total UTs = Num_Sectors * Num_UT_Per_Sector
        # We assume 1 BS = 1 Sector for this loader context usually,
        # unless orientations imply sectors.
        # Let's assume generic "Transmitters" in file correspond to logical sectors/cells.

        num_ut_total = self.num_bs * self.config.num_ut_per_sector

        # Select random mesh points
        # mesh_locs: [N_UT, 3] (Local coords relative to origin if loader handles conversion, or straight from mesh which is UTM?)
        # MeshBasedLoader.get_random_mesh_coordinates returns LOCAL coordinates (relative to origin).
        # Wait, check loader implementation:
        # returns self._mesh_local[indices] -> _mesh_local is utm_to_local(mesh_utm).
        # So it returns Local coords. Good.

        # But wait, get_topology needs to return consistent coordinates.
        # If BS locs are UTM, UT locs should be UTM? Or both Local?
        # Simulation usually runs in Local Cartesian.
        # We should provide Local coordinates.

        # Convert cached BS UTM to Local
        # loader.geo is the CoordinateSystem
        self.bs_loc_local = self.loader.geo.utm_to_local(self.bs_loc_utm)

        # Get random UTs
        # If we want to ensure "per sector" distribution, we should use get_random_coordinates_by_best_server
        # But determining best server requires path loss. MeshLoader has get_best_server_mapping().

        ut_locs = []
        ut_indices = []

        # Try to distribute UEs per BS if possible
        try:
            # This requires 'path_gain' or 'pathloss' to be present and mappable in loader
            for bs_i in range(self.num_bs):
                # Get N points for this BS
                # We need the indices to create ExternalPaths later!
                # loader.get_random_coordinates_by_best_server returns coords, not indices.
                # We should probably extend loader or implement bespoke logic here.
                # For now, let's just pick random points globally if we can't easily do per-sector.
                # Or meaningful drop logic:
                pass

        except Exception:
            # Fallback to random global distribution
            pass

        # Simple Global Random for now (or implement indices retrieval)
        # MeshBasedLoader doesn't strictly expose "get_indices_for_random".
        # Let's allow duplicates? No.

        # Creating a custom random selector that returns indices
        num_mesh = self.loader._mesh_local.shape[0]
        if num_mesh < num_ut_total:
            self.logger.warning(
                f"Not enough mesh points ({num_mesh}) for {num_ut_total} UTs. Enabling replacement."
            )
            replace = True
        else:
            replace = False

        # Uniform random selection
        selected_indices = np.random.choice(
            num_mesh, size=num_ut_total, replace=replace
        )

        self.current_ut_idx = selected_indices
        self.current_ut_loc_local = self.loader._mesh_local[
            selected_indices
        ]  # [N_UT, 3]

    def find_nearest_mesh(self, ut_loc):
        """
        Existing interface uses this?
        If we load from external, ut_loc is derived FROM the mesh, so we already have indices.
        But simulator might call this if it generates topology internally.
        Since we provide topology, this might be redundant but good to have.
        """
        # Simulator relies on this to map continuous coords to mesh points.
        # But here we dictate the coords.
        return self.current_ut_idx

    def get_topology(self):
        """
        Returns the dictionary of topology tensors required by SystemSimulator.
        """
        if self.current_ut_idx is None:
            self.load_drop(0)  # Initial drop

        # Prepare Tensors [Batch, N, 3]
        # Batch size is handled by Simulator setup, but get_topology usually returns [Batch, ...] ?
        # Looking at runner.py/simulator.py:
        # _setup_topology call gen_hexgrid which returns [Batch, ...].
        # So we must broadcast to batch size.
        B = self.config.batch_size

        ut_loc = tf.convert_to_tensor(
            self.current_ut_loc_local, dtype=tf.float32
        )  # [N_UT, 3]
        bs_loc = tf.convert_to_tensor(self.bs_loc_local, dtype=tf.float32)  # [N_BS, 3]

        # Broadcast
        ut_loc = tf.broadcast_to(ut_loc[None, ...], [B, *ut_loc.shape])
        bs_loc = tf.broadcast_to(bs_loc[None, ...], [B, *bs_loc.shape])

        # Orientations
        # BS: Fixed from file
        bs_orient = tf.convert_to_tensor(self.bs_orient, dtype=tf.float32)
        bs_orient = tf.broadcast_to(bs_orient[None, ...], [B, *bs_orient.shape])

        # UT: Random or fixed?
        # Random yaw
        num_ut = ut_loc.shape[1]
        ut_orient_val = np.zeros((num_ut, 3), dtype=np.float32)
        ut_orient_val[:, 0] = np.random.uniform(
            0, 2 * np.pi, size=num_ut
        )  # Radians? or Degrees?
        # Sionna usually expects Radians for calculation but generic inputs might be deg?
        # gen_hexgrid returns [Batch, NumUT, 3].
        # Let's assume Radians for internal math? Wrapper converts?
        # wsim/common/geo uses radians usually?
        # Wait, Sionna internal is radians. gen_hexgrid?
        # Check standard sionna... usually radians.
        ut_orient = tf.convert_to_tensor(ut_orient_val, dtype=tf.float32)
        ut_orient = tf.broadcast_to(ut_orient[None, ...], [B, *ut_orient.shape])

        # Velocities: Zero for static
        ut_vel = tf.zeros_like(ut_loc)

        # In State: All Indoor/Outdoor?
        # External data might specify? Assuming Outdoor (False) or generic.
        in_state = tf.zeros((B, num_ut), dtype=tf.bool)  # False = Outdoor?

        # LoS: From map if available.
        # If we don't have LoS map, we can return None or True?
        # Used for channel model selection (if statistical).
        # Since we use RT, this is less critical unless Hybrid needs it.
        los = tf.zeros((B, num_ut, self.num_bs), dtype=tf.bool)  # Dummy

        topo_dict = {
            "ut_loc": ut_loc,
            "bs_loc": bs_loc,
            "ut_orient": ut_orient,
            "bs_orient": bs_orient,
            "ut_vel": ut_vel,
            "in_state": in_state,
            "los": los,
            # "serving_cell_id": ... # Optional, can be derived by distance
        }

        if self.config.topology_wrap:
            # If wrap is enabled, try to provide bs_virtual_loc from the loader's metadata
            # For bounded meshes without wrap, this will remain None or we can fallback to bs_loc
            if (
                hasattr(self.loader, "bs_virtual_loc")
                and self.loader.bs_virtual_loc is not None
            ):
                bs_v_loc = tf.convert_to_tensor(
                    self.loader.bs_virtual_loc, dtype=tf.float32
                )
                bs_v_loc = tf.broadcast_to(bs_v_loc[None, ...], [B, *bs_v_loc.shape])
                topo_dict["bs_virtual_loc"] = bs_v_loc
            else:
                # Provide a dummy single-wrap (standard positions) as fallback
                topo_dict["bs_virtual_loc"] = tf.expand_dims(bs_loc, axis=2)

        return topo_dict

    def __call__(self, config):
        """
        Factory method to return the channel model instance?
        Or ExternalPaths?

        SystemSimulator calls: self.channel_model = external_loader(config)
        HybridChannelInterface calls: external_loader.get_paths(...) ?
        Or channel_model.generate()?

        In Simulator.py:
          if external_loader is not None:
             self.channel_model = external_loader(config)

        AND

          self.channel_interface = HybridChannelInterface(..., external_loader=self.external_loader)

        It seems 'external_loader' argument to Simulator is the CLASS or FACTORY, which returns 'channel_model'.
        But HybridChannelInterface also takes 'external_loader' instance to call 'get_paths'?

        We need to align this.
        If we pass THIS instance as `external_loader`, we can make it callable to return... itself?
        Or a dummy object that behaves like a channel model?

        Actually, HybridChannelInterface uses `channel_model` primarily for statistical generation.
        If we strictly use RT, `channel_model` might be unused or just a placeholder.

        Let's allow this instance to be passed as is.
        """
        return self

    def get_rays(self, ut_indices, bs_indices=None):
        """
        Interface method called by HybridChannelInterface to retrieve path data.
        Delegates to MeshBasedLoader.get_paths using indices.

        Args:
            ut_indices (tf.Tensor or np.ndarray): Indices of UTs in the mesh key.
            bs_indices: Ignored for now (returns all BS paths typically).
        """
        if isinstance(ut_indices, tf.Tensor):
            ut_indices = ut_indices.numpy()

        # Flatten if necessary as get_paths expects 1D indices for [N] list
        # But if it's batched [Batch, N_UT], we need to be careful.
        # MeshBasedLoader returns Paths with shape [num_rx, num_tx, ...].
        # If we pass M indices, we get M RXs.
        flat_indices = ut_indices.flatten()

        # Call loader
        # Ensure integer type
        flat_indices = flat_indices.astype(np.int32)
        paths = self.loader.get_paths(flat_indices)

        # Convert ExternalPaths to dictionary expected by HybridChannelInterface
        # The interface expects keys:
        # - rays: delays, powers, aoa, aod, zoa, zod, xpr
        # - LSPs: pathloss, shadow_fading, k_factor

        a = paths.a

        if isinstance(a, tuple):
            if len(a) == 2:
                # Power = real^2 + imag^2
                powers = dr.sqr(a[0]) + dr.sqr(a[1])
                # Return tuple as is for path_gains
                path_gains = a
            else:
                raise ValueError(f"Unexpected tuple length for paths.a: {len(a)}")
        else:
            powers = dr.sqr(dr.abs(a))
            path_gains = a

        # Create dummy LSPs
        # pathloss=0 (dB), shadow_fading=1 (linear), k_factor=0 (linear)
        # We need shapes. paths.tau is [Num_RX, Num_TX, Max_Paths]
        # LSPs should be [Num_RX, Num_TX] (or similar depending on interface expectation)
        # DrJit arrays don't expose .shape easily if dynamic, but we can infer from len?
        # Actually Sionna Paths usually have .shape attribute if wrapped?
        # Or we can use dr.width(paths.tau) but that is total elements.

        # However, for hybrid.py, these come from data[...] which are expected to be Tensors/Numpy arrays?
        # The hybrid interface calls gather on them.
        # If we return DrJit arrays, will tf.gather work?
        # Sionna's Rays usually holds Tensor or DrJit arrays.
        # The gather_neighbor_data function in hybrid.py uses tf.gather.
        # tf.gather works on Tensors. It might fail on DrJit arrays depending on interop.
        # It is safer to convert everything to TensorFlow tensors or NumPy arrays here.

    def get_rays(self, ut_indices, bs_indices=None):
        # Fetch rays from underlying loader
        paths = self.loader.get_paths(ut_indices)

        # Convert ExternalPaths to dictionary expected by HybridChannelInterface
        path_gains = paths.a
        powers = paths.powers

        def to_tf(x, is_angle=False):
            # Convert DrJit array to TF tensor
            if isinstance(x, tuple):
                return tf.complex(to_tf(x[0], is_angle), to_tf(x[1], is_angle))

            t = tf.convert_to_tensor(np.array(x), dtype=tf.float32)
            # Squeeze antenna dims if they are singleton [RX, 1, TX, 1, ...] -> [RX, TX, ...]
            # ExternalPaths produces [RX, 1, TX, 1, ...] for delay/powers/angles
            if len(t.shape) >= 4 and t.shape[1] == 1 and t.shape[3] == 1:
                t = tf.squeeze(t, axis=[1, 3])

            # Now t is [RX, TX, ...]
            # Swap RX and TX dimensions: [TX, RX, ...]
            rank = tf.rank(t)
            perm = tf.concat([[1, 0], tf.range(2, rank)], axis=0)
            t = tf.transpose(t, perm)

            if is_angle:
                # Add Ray dimension: [TX, RX, Clusters] -> [TX, RX, Clusters, 1]
                t = tf.expand_dims(t, -1)

            # Add batch dimension [1, ...]
            return tf.expand_dims(t, 0)

        delays = to_tf(paths.tau, is_angle=False)  # [1, TX, RX, Clusters]

        # Shape handling for LSPs
        shape_full = delays.shape
        num_tx = shape_full[1]
        num_rx = shape_full[2]

        # LSPs: [1, TX, RX]
        pl = tf.zeros((1, num_tx, num_rx), dtype=tf.float32)
        sf = tf.ones((1, num_tx, num_rx), dtype=tf.float32)
        kf = tf.zeros((1, num_tx, num_rx), dtype=tf.float32)

        # XPR: [1, TX, RX, Clusters, 1]
        xpr = tf.zeros_like(to_tf(paths.tau, is_angle=True))

        # Helper for path_gains
        if isinstance(path_gains, tuple):
            pg_tf = tf.complex(
                to_tf(path_gains[0], is_angle=True), to_tf(path_gains[1], is_angle=True)
            )
        else:
            pg_tf = to_tf(path_gains, is_angle=True)

        return {
            "delays": delays,
            "path_gains": pg_tf,
            "powers": to_tf(powers, is_angle=False),
            "aod": to_tf(paths.phi_t, is_angle=True),
            "zod": to_tf(paths.theta_t, is_angle=True),
            "aoa": to_tf(paths.phi_r, is_angle=True),
            "zoa": to_tf(paths.theta_r, is_angle=True),
            "xpr": xpr,
            "pathloss": pl,
            "shadow_fading": sf,
            "k_factor": kf,
        }

    def set_topology(self, *args, **kwargs):
        """
        Dummy method to satisfy SystemSimulator interface.
        Topology is managed internally by this loader.
        """
        pass

    def get_power_map(self, ut_indices):
        """
        Returns received power map [Batch, UT, BS] in dBm.
        Assumes BS transmits at config.bs_max_power_dbm.

        Args:
            ut_indices: Indices of UTs in the mesh (or dummy indices if loading logic handles it)
                        Expected shape: [Batch, UT] or [UT]
        """
        # Ensure indices are handled correctly (Batch dimension might be present)
        # get_rays handles flattening internally if tensor is passed
        ray_data = self.get_rays(ut_indices)

        # powers: [Batch, TX, RX, Paths] (linear power per path)
        # Note: get_rays returns 'powers' as [Batch, TX, RX, Paths] based on my implementation
        # Checks step 741 code:
        # returns "powers": to_tf(powers, is_angle=False)
        # to_tf: expands dims, then:
        # [RX, 1, TX, 1, Paths] -> [RX, TX, Paths] (squeeze 1,3)
        # -> [TX, RX, Paths] (transpose)
        # -> [1, TX, RX, Paths] (expand dims 0)

        powers_linear = ray_data["powers"]  # [Batch, TX, RX, Paths]

        # Sum over paths to get total channel gain (linear)
        # [Batch, TX, RX]
        channel_gain_linear = tf.reduce_sum(powers_linear, axis=-1)

        # Transpose to [Batch, RX, TX] as expected by simulator (UT, BS)
        channel_gain_linear = tf.transpose(channel_gain_linear, perm=[0, 2, 1])

        # Convert to dB
        # Avoid log(0)
        channel_gain_db = (
            10.0
            * tf.math.log(tf.maximum(channel_gain_linear, 1e-20))
            / tf.math.log(10.0)
        )

        # Add TX Power (BS Max Power)
        # Result = P_TX + Gain = P_RX
        # Broadcast bs_max_power_dbm
        powers_dbm = self.config.bs_max_power_dbm + channel_gain_db

        return powers_dbm
