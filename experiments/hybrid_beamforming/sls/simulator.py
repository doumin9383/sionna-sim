import sionna
import tensorflow as tf

# Additional external libraries
import matplotlib.pyplot as plt
import numpy as np

# Sionna components
from sionna.sys import gen_hexgrid_topology, get_num_hex_in_grid
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.utils import dbm_to_watt
from sionna.phy import Block
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, PanelArray
from sionna.phy.ofdm import ResourceGrid

# Local Components
from .components.hybrid_channel_interface import HybridChannelInterface
from .components.mpr_model import MPRModel
from .components.power_control import PowerControl
from .components.link_adaptation import MCSLinkAdaptation
from .components.get_hist import init_result_history, record_results
from .components.precoder_utils import expand_precoder
from .components.beam_management import BeamSelector

# Set random seed for reproducibility
sionna.phy.config.seed = 42

# Internal computational precision
sionna.phy.config.precision = "single"  # 'single' or 'double'


from .configs import SLSConfig


class SystemSimulator(Block):

    def __init__(
        self,
        config: SLSConfig,
        max_bs_ut_dist=None,
        min_bs_ut_dist=None,
        temperature=294,
        o2i_model="low",
        average_street_width=20.0,
        average_building_height=5.0,
        precision=None,
        external_loader=None,
    ):
        super().__init__(precision=precision)

        self.config = config
        self.external_loader = external_loader
        self.scenario = config.scenario
        self.batch_size = int(config.batch_size)
        self.coherence_time = tf.cast(config.coherence_time, tf.int32)  # [slots]
        # num_cells = get_num_hex_in_grid(config.num_rings) # Moved to config
        # self.num_bs = num_cells * 3 # Moved to config
        self.num_bs = config.num_bs
        self.num_ut_per_sector = int(config.num_ut_per_sector)
        self.direction = config.direction
        self.bs_max_power_dbm = config.bs_max_power_dbm
        self.ut_max_power_dbm = config.ut_max_power_dbm
        self.num_ut = self.num_bs * self.num_ut_per_sector

        # Antenna Counts
        self.num_bs_ant = config.bs_array.num_ant
        self.num_ut_ant = config.ut_array.num_ant

        # Instantiate ResourceGrid from config (Factory)
        self.resource_grid = ResourceGrid(
            # num_ofdm_symbols=config.num_ofdm_symbols,
            num_ofdm_symbols=1,  # チャネル生成のためだけに使うので1でOK
            fft_size=self.config.num_subcarriers,
            subcarrier_spacing=self.config.subcarrier_spacing,
            # pilot_pattern=rg_config.pilot_pattern,
            # pilot_ofdm_symbol_indices=rg_config.pilot_ofdm_symbol_indices,
        )

        if self.direction == "uplink":
            self.num_tx, self.num_rx = self.num_ut, self.num_bs
            self.num_tx_per_sector = self.num_ut_per_sector
        else:
            self.num_tx, self.num_rx = self.num_bs, self.num_ut
            self.num_tx_per_sector = 1

        # Precoding Granularity Settings
        self.precoding_granularity = config.precoding_granularity
        self.rbg_size_rb = config.rbg_size_rb
        self.rbg_size_sc = int(self.rbg_size_rb * 12) if self.rbg_size_rb > 0 else None

        # Noise power per subcarrier
        self.no = tf.cast(
            BOLTZMANN_CONSTANT * temperature * config.subcarrier_spacing,
            self.rdtype,
        )

        # Slot duration [sec]
        self.slot_duration = (
            self.resource_grid.ofdm_symbol_duration
            * self.resource_grid.num_ofdm_symbols
        )

        # Initialize channel model based on scenario
        if external_loader is not None:
            self.channel_model = external_loader(config)
        else:
            self._setup_channel_model(
                config.scenario,
                config.carrier_frequency,
                o2i_model,
                self.config.ut_array,
                self.config.bs_array,
                average_street_width,
                average_building_height,
            )

        # Generate multicell topology
        # Moved to call method loop for drops
        # self._setup_topology(config.num_rings, min_bs_ut_dist, max_bs_ut_dist)

        # Instantiate the Hybrid Channel Interface
        if self.direction == "uplink":
            tx_array = self.config.ut_array
            rx_array = self.config.bs_array

            # Calculate ports fallback
            ut_pol_factor = 2 if self.config.ut_polarization == "dual" else 1
            num_tx_ports = (
                self.config.ut_num_rows_panel
                * self.config.ut_num_cols_panel
                * ut_pol_factor
            )

            bs_pol_factor = 2 if self.config.bs_polarization == "dual" else 1
            num_rx_ports = (
                self.config.bs_num_rows_panel
                * self.config.bs_num_cols_panel
                * bs_pol_factor
            )

        else:
            tx_array = self.config.bs_array
            rx_array = self.config.ut_array

            bs_pol_factor = 2 if self.config.bs_polarization == "dual" else 1
            num_tx_ports = (
                self.config.bs_num_rows_panel
                * self.config.bs_num_cols_panel
                * bs_pol_factor
            )

            ut_pol_factor = 2 if self.config.ut_polarization == "dual" else 1
            num_rx_ports = (
                self.config.ut_num_rows_panel
                * self.config.ut_num_cols_panel
                * ut_pol_factor
            )

        self.num_tx_ports = num_tx_ports
        self.num_rx_ports = num_rx_ports

        self.channel_interface = HybridChannelInterface(
            channel_model=self.channel_model,
            resource_grid=self.resource_grid,
            tx_array=tx_array,
            rx_array=rx_array,
            num_tx_ports=self.num_tx_ports,
            num_rx_ports=self.num_rx_ports,
            precision=self.precision,
            use_rbg_granularity=config.use_rbg_granularity,
            rbg_size_sc=self.rbg_size_sc if self.rbg_size_sc else 1,
            neighbor_indices=None,  # Topology is set in the loop and updated dynamically
            external_loader=self.external_loader,
        )

        # Instantiate SLS components
        self.mpr_model = MPRModel(csv_path=config.mpr_table_path)
        self.power_control = PowerControl(p_power_class=config.ut_max_power_dbm)
        self.mcs_adapter = MCSLinkAdaptation()

        # Instantiate Beam Manager
        self.beam_selector = BeamSelector(
            num_rows_per_panel=config.bs_num_rows_per_panel,
            num_cols_per_panel=config.bs_num_cols_per_panel,
            num_panels_v=config.bs_num_rows_panel,
            num_panels_h=config.bs_num_cols_panel,
            polarization=config.bs_polarization,
            oversampling_factor=config.beambook_oversampling_factor,
            dtype=self.rdtype,  # Use simulation precision
        )

    def _setup_channel_model(
        self,
        scenario,
        carrier_frequency,
        o2i_model,
        ut_array,
        bs_array,
        average_street_width,
        average_building_height,
    ):
        """Initialize appropriate channel model based on scenario"""
        common_params = {
            "carrier_frequency": carrier_frequency,
            "ut_array": ut_array,
            "bs_array": bs_array,
            "direction": self.direction,
            "enable_pathloss": True,
            "enable_shadow_fading": True,
            "precision": self.precision,
        }

        if scenario == "umi":  # Urban micro-cell
            self.channel_model = UMi(o2i_model=o2i_model, **common_params)
        elif scenario == "uma":  # Urban macro-cell
            self.channel_model = UMa(o2i_model=o2i_model, **common_params)
        elif scenario == "rma":  # Rural macro-cell
            self.channel_model = RMa(
                average_street_width=average_street_width,
                average_building_height=average_building_height,
                **common_params,
            )

    def _setup_topology(self, num_rings, min_bs_ut_dist, max_bs_ut_dist):
        """Generate and set up network topology"""

        # 0. Load Topology from External Loader if available
        if self.external_loader is not None:
            topo = self.external_loader.get_topology()
            self.ut_loc = topo["ut_loc"]
            self.bs_loc = topo["bs_loc"]
            self.ut_orientations = topo["ut_orient"]
            self.bs_orientations = topo["bs_orient"]
            self.ut_velocities = topo["ut_vel"]
            self.in_state = topo["in_state"]
            self.los = topo["los"]

            if "serving_cell_id" in topo:
                self.serving_cell_id = topo["serving_cell_id"]
            else:
                self.serving_cell_id = None

            self.ut_mesh_indices = self.external_loader.find_nearest_mesh(self.ut_loc)

            # BS Virtual Loc needs to be derived or loaded?
            # gen_hexgrid_topology returns it.
            # If not saved, we might need it for some logic?
            # It's used for wrapped-around distance calculation if enabled.
            # If we don't support wrap-around in external mode or if saved bs_loc is already virtual/final?
            # Usually bs_loc is enough for distance if no wrap-around logic is explicitly called later.
            # We'll set it to None or bs_loc for now.
            self.bs_virtual_loc = None

            # Update counts to match external data
            self.num_ut = tf.shape(self.ut_loc)[1]
            self.num_bs = tf.shape(self.bs_loc)[1]
            # num_ut_per_sector might be affected if we assume it's fixed,
            # but usually for external data we treat it as 1 or adjust it.
            # To be safe, let's keep consistency.

        elif self.config.topology_type == "HexGrid":
            (
                self.ut_loc,
                self.bs_loc,
                self.ut_orientations,
                self.bs_orientations,
                self.ut_velocities,
                self.in_state,
                self.los,
                self.bs_virtual_loc,
                self.grid,
            ) = gen_hexgrid_topology(
                batch_size=self.batch_size,
                num_rings=num_rings,
                num_ut_per_sector=self.num_ut_per_sector,
                min_bs_ut_dist=min_bs_ut_dist,
                max_bs_ut_dist=max_bs_ut_dist,
                scenario=self.scenario,
                los=True,
                return_grid=True,
                precision=self.precision,
            )
            self.serving_cell_id = None
        else:
            raise NotImplementedError(
                f"Topology type {self.config.topology_type} not implemented in _setup_topology"
            )

        # 1. Calculate distances [batch, num_ut, num_bs]
        # ut_loc: [batch, num_ut, 3], bs_loc: [batch, num_bs, 3]
        diff = tf.expand_dims(self.ut_loc, axis=2) - tf.expand_dims(self.bs_loc, axis=1)
        dist = tf.norm(diff, axis=-1)

        # 2. Define Explicit Serving BS Association
        if self.serving_cell_id is not None:
            # Use external serving cell ID if available
            # Ensure shape is [batch, num_ut]
            self.serving_bs_ids = self.serving_cell_id
        else:
            # Fallback (HexGrid): We enforce a strict mapping: UEs are associated with BSs sequentially.
            # This matches the ID-based logic used in beamforming and resource allocation.
            ut_ids = tf.range(self.num_ut)
            serving_bs_ids_flat = ut_ids // self.num_ut_per_sector

            # Store as class member for use in other methods
            self.serving_bs_ids = tf.broadcast_to(
                serving_bs_ids_flat[None, :], [self.batch_size, self.num_ut]
            )

        # 3. Select Neighbors: Serving BS (Index 0) + Top K-1 Interferers

        # get actual shapes from data
        current_num_ut = tf.shape(dist)[1]
        current_num_bs = tf.shape(dist)[2]

        # Create a mask for the serving BS
        bs_range = tf.range(current_num_bs)
        # [batch, num_ut, num_bs]
        is_serving = (
            self.serving_bs_ids[:, :current_num_ut, None] == bs_range[None, None, :]
        )

        # Modify distance for sorting:
        # Assign a large negative 'distance' to the Serving BS so it is always selected first
        # by top_k(-dist) which prioritizes smallest distances (largest negative values).
        dist_modified = tf.where(is_serving, -1.0e9, dist)

        # Select neighbors
        _, neighbor_indices = tf.math.top_k(-dist_modified, k=self.config.num_neighbors)
        # neighbor_indices shape: [batch, num_ut, num_neighbors]
        self.neighbor_indices = neighbor_indices

        # 3. Set topology in channel model

        # Determine LoS argument
        if self.external_loader is not None:
            # Pass None for 'los' to avoid validation error/re-generation when using external Rays
            los_arg = None
        else:
            los_arg = self.los

        self.channel_model.set_topology(
            self.ut_loc,
            self.bs_loc,
            self.ut_orientations,
            self.bs_orientations,
            self.ut_velocities,
            self.in_state,
            los_arg,
            self.bs_virtual_loc,
        )

    # @tf.function(jit_compile=False)
    def call(self, num_drops, tx_power_dbm):
        # We use 'num_drops' as the time dimension in the history
        hist = init_result_history(
            self.batch_size, num_drops, self.num_bs, self.num_ut_per_sector
        )

        # --------------- #
        # Simulate Drops  #
        # --------------- #
        # We use a Python loop instead of tf.while_loop to allow
        # explicit topology reset and graph re-execution/eager execution for each drop.
        # This helps in managing memory and ensures topology is actually updated.

        for drop_idx in range(num_drops):
            # 0. Set up new topology for this drop
            if self.external_loader is not None:
                self.external_loader.load_drop(drop_idx)

            # This generates new UT locations, channels, etc.
            # If external_loader is used, _setup_topology should use it.
            self._setup_topology(
                self.config.num_rings,
                self.config.min_bs_ut_dist,
                self.config.max_bs_ut_dist,
            )

            # 1. Get Channel Information (Batched to avoid OOM)
            # We only need h_serv (Serving Link) for Beam Selection.
            # h_elem_all (including neighbors) is too large to hold in memory for all UTs at once.

            h_serv_list = []
            batch_size_beam = 1  # Process 1 UT at a time to be safe

            for i in range(0, self.num_ut, batch_size_beam):
                end_i = min(i + batch_size_beam, self.num_ut)

                # Slice Topology for this batch
                # neighbor_indices: [Batch, Num_UT, Neighbor]
                curr_neigh_inds = self.neighbor_indices[:, i:end_i, :]

                # Slice UT properties
                curr_ut_loc = self.ut_loc[:, i:end_i, :]
                curr_ut_orient = self.ut_orientations[:, i:end_i, :]
                curr_ut_vel = self.ut_velocities[:, i:end_i, :]
                curr_in_state = self.in_state[:, i:end_i]

                # Get Element Channel for this batch
                h_elem_batch = (
                    self.channel_interface.get_element_channel_for_beam_selection(
                        batch_size=self.batch_size,
                        ut_loc=curr_ut_loc,
                        bs_loc=self.bs_loc,
                        ut_orient=curr_ut_orient,
                        bs_orient=self.bs_orientations,
                        neighbor_indices=curr_neigh_inds,
                        ut_velocities=curr_ut_vel,
                        in_state=curr_in_state,
                    )
                )

                # Extract Serving Channel (Neighbor 0) and remove Time dimension (index 5)
                # h_elem_batch: [Batch, Batch_UT, Neighbors, RxA, TxA, Time, SC] (Wait, check order)
                # Order confirmed: [Batch, UT, Neighbor, RxA, TxA, Time, Freq]
                # We want Neighbor=0, Time=0.
                h_serv_batch = h_elem_batch[:, :, 0, :, :, 0, :]
                h_serv_list.append(h_serv_batch)

            # Concatenate all batches
            # h_serv: [Batch, Total_UT, SC, RxA, TxA] (assuming Time=1 and squeezed)
            # wait, h_elem_batch[:, :, 0, 0, :, :, :] removes Neighbor(2) and Time(3).
            # Result dims: [Batch, Batch_UT, SC, RxA, TxA]
            h_serv = tf.concat(h_serv_list, axis=1)  # Concat along UT axis

            # 2. Select Analog Beams (BS Side)
            # h_serv channel is from UT to Serving BS (because neighbor_index=0 is serving BS)
            # We need to restructure this into [Batch, Num_BS, Num_UT_Per_Sector, SC, RxA, TxA]
            # using the serving_bs_ids map.

            # Dimensions
            B = self.batch_size
            N_BS = self.num_bs
            N_UT_Sec = self.num_ut_per_sector  # Should be 1 typically
            N_UT_Total = self.num_ut
            # h_serv shape: [Batch, Total_UT, RxA, TxA, Freq] (Verified)
            RxA = tf.shape(h_serv)[2]
            TxA = tf.shape(h_serv)[3]
            FFT = tf.shape(h_serv)[4]

            # Construct indices for scatter update
            # Indices: [b, ut_idx] -> [b, serving_bs_id, slot_id]

            batch_indices = tf.range(B)[:, None]  # [B, 1]
            batch_indices = tf.broadcast_to(
                batch_indices, [B, N_UT_Total]
            )  # [B, Num_UT]

            bs_indices = tf.cast(self.serving_bs_ids, tf.int32)  # [B, Num_UT]

            # num_freq_points is usually num_rbgs for beam selection
            num_freq_points = tf.shape(h_serv)[-1]

            # scatter_nd sums duplicate updates.
            # We assume N_UT_Sec=1 and each BS is served by at most 1 UE for this logic.
            slot_indices = tf.zeros_like(bs_indices)

            # Stack to create indices: [B, Num_UT, 3] -> (Batch, BS, Slot)
            indices = tf.stack([batch_indices, bs_indices, slot_indices], axis=-1)

            # Prepare Updates (h_serv)
            # h_serv: [B, BUT, 1, RxA, TxA, 1, Freq] -> [B, BUT, Features]
            # Flatten everything after BUT
            h_serv_flat = tf.reshape(h_serv, [B, N_UT_Total, -1])

            # Prepare Canvas (h_bs)
            feature_dim = RxA * TxA * num_freq_points
            h_bs_shape = [B, N_BS, N_UT_Sec, feature_dim]

            # Scatter
            h_bs_flat = tf.scatter_nd(indices, h_serv_flat, h_bs_shape)

            # Reshape back to detailed dimensions
            h_bs = tf.reshape(h_bs_flat, [B, N_BS, N_UT_Sec, RxA, TxA, num_freq_points])

            # Note: scatter_nd fills unassigned slots with 0.

            if self.direction == "uplink":
                # BS is Receiver: RxA = num_bs_ant, TxA = num_ut_ant
                # h_bs: [B, num_bs, ut, BS_Ant, UT_Ant, SC]
                # Target: [Batch, num_bs, num_ut_per_sector, UT_Ant, BS_Ant, SC]
                # Indices: 0, 1, 2, 4(UT), 3(BS), 5(SC)
                h_bs_permuted = tf.transpose(h_bs, [0, 1, 2, 4, 3, 5])
            else:
                # BS is Transmitter: RxA = num_ut_ant, TxA = num_bs_ant
                # h_bs: [B, num_bs, ut, UT_Ant, BS_Ant, SC]
                # Target: [Batch, num_bs, num_ut_per_sector, UT_Ant, BS_Ant, SC]
                # Indices: 0, 1, 2, 3(UT), 4(BS), 5(SC)
                h_bs_permuted = tf.transpose(h_bs, [0, 1, 2, 3, 4, 5])

            # Flatten to [Batch * num_bs, num_ut_per_sector, Other_Ant, BS_Ant, SC]
            h_selector_input = tf.reshape(
                h_bs_permuted,
                [
                    -1,
                    self.num_ut_per_sector,
                    tf.shape(h_bs_permuted)[3],
                    self.num_bs_ant,
                    tf.shape(h_bs_permuted)[5],
                ],
            )

            # BS Beam Selection
            # w_rf_bs_flat: [Batch * num_bs, TotalAnt, TotalPorts]
            w_rf_bs_flat = self.beam_selector(h_selector_input, self.config.bs_array)

            # Reshape back to [Batch, num_bs, TotalAnt, TotalPorts]
            w_rf_bs = tf.reshape(
                w_rf_bs_flat, [self.batch_size, self.num_bs, self.num_bs_ant, -1]
            )

            # UE Analog Beam: Identity (Full Digital or Fixed)
            if self.direction == "uplink":
                ue_ports = self.channel_interface.hybrid_channel.num_tx_ports
            else:
                ue_ports = self.channel_interface.hybrid_channel.num_rx_ports

            a_rf_ue = tf.eye(self.num_ut_ant, num_columns=ue_ports, dtype=tf.complex64)

            # Apply weights
            if self.direction == "uplink":
                self.channel_interface.set_analog_weights(w_rf=a_rf_ue, a_rf=w_rf_bs)
                u_bs_global = w_rf_bs
                v_bs_global = None
            else:
                self.channel_interface.set_analog_weights(w_rf=w_rf_bs, a_rf=a_rf_ue)
                v_bs_global = w_rf_bs
                u_bs_global = None

            # --- 3. Power Control & Link Adaptation (Moved up) ---
            # For Power Control: Identify Serving BS
            serving_bs_idx = self.neighbor_indices[:, :, 0]
            serving_bs_idx_i32 = tf.cast(serving_bs_idx, tf.int32)

            serving_bs_idx_batched = tf.broadcast_to(
                serving_bs_idx_i32, [self.batch_size, self.num_ut]
            )
            serving_bs_loc = tf.gather(
                self.bs_loc, serving_bs_idx_batched, axis=1, batch_dims=1
            )
            dist = tf.norm(self.ut_loc - serving_bs_loc, axis=-1)  # [batch, num_ut]

            # Simple UMi Path Loss Model for 3.5GHz (Placeholder or External)
            if self.external_loader is not None:
                powers_dbm = self.external_loader.get_power_map(self.ut_mesh_indices)
                # For interference, we use powers directly later,
                # but path loss to serving cell is needed for Power Control
                serving_bs_idx_expand = tf.expand_dims(serving_bs_idx_batched, axis=-1)
                serving_power = tf.gather(
                    powers_dbm, serving_bs_idx_expand, axis=2, batch_dims=2
                )
                serving_power = tf.squeeze(serving_power, axis=-1)
                pl_db = self.config.bs_max_power_dbm - serving_power
            else:
                fc_ghz = 3.5
                dist_safe = tf.maximum(dist, 1.0)
                pl_db = (
                    28.0
                    + 22.0 * tf.math.log(dist_safe) / tf.math.log(10.0)
                    + 20.0 * tf.math.log(fc_ghz) / tf.math.log(10.0)
                )

            # b. Get MPR
            # Assuming "CP-OFDM" and Rank 1 for simplified PC
            # In future, use actual scheduler rank
            mpr_val = self.mpr_model.get_mpr("CP-OFDM", 1)  # Scalar approximation
            mpr_db = tf.fill(
                [self.batch_size, self.num_ut], tf.cast(mpr_val, tf.float32)
            )

            # c. Calculate Tx Power
            if self.direction == "uplink":
                # num_rbs: Total RBs (assuming full bw allocation for now or partial)
                # resource_grid.num_effective_subcarriers / 12
                num_rbs = self.resource_grid.num_effective_subcarriers / 12.0
                p_tx_dbm = self.power_control.calculate_tx_power(
                    pl_db, num_rbs, mpr_val
                )
                p_cmax_dbm = self.ut_max_power_dbm - mpr_val
            else:
                # Downlink: Use fixed power
                p_tx_dbm = tx_power_dbm
                p_cmax_dbm = self.bs_max_power_dbm  # Assuming no MPR for BS

            p_cmax_dbm_tensor = tf.fill(
                [self.batch_size, self.num_ut], tf.cast(p_cmax_dbm, tf.float32)
            )

            # Calculate Total Power and Allocation per stream
            total_power = dbm_to_watt(p_tx_dbm)
            # Reshape/Broadcast for broadcasting: [batch, num_ut, 1, 1, 1]
            if len(total_power.shape) == 0:  # Scalar
                total_power = tf.broadcast_to(
                    total_power, [self.batch_size, self.num_ut]
                )

            # p_alloc setup will happen after Rank determination

            # --- 4. Effective Channel & SINR Calculation (BATCHED to avoid OOM) ---
            # batch_size_ut_sinr = 1 is safest for 718MB VRAM
            batch_size_ut_sinr = 1
            num_batches_ut_sinr = (
                self.num_ut + batch_size_ut_sinr - 1
            ) // batch_size_ut_sinr

            # Constants for SINR Calculation
            B = self.batch_size
            N_BS = self.num_bs
            N_UT_Total = self.num_ut
            FFT = self.resource_grid.fft_size
            rank = 1

            # Power Allocation (based on Rank)
            p_alloc = total_power / tf.cast(rank, self.rdtype)
            p_alloc = tf.reshape(p_alloc, [B, N_UT_Total, 1, 1])

            # Buffers for results
            s_power_all = []
            i_total_all = []

            # Digital weights buffers [B, Entity, Freq, Ports, Rank]
            # UT-side (Terminal)
            w_ut_dig = tf.zeros(
                [
                    B,
                    N_UT_Total,
                    FFT,
                    (
                        self.num_tx_ports
                        if self.direction == "uplink"
                        else self.num_rx_ports
                    ),
                    rank,
                ],
                dtype=self.cdtype,
            )
            # BS-side (Sector)
            w_bs_dig = tf.zeros(
                [
                    B,
                    N_BS,
                    FFT,
                    (
                        self.num_rx_ports
                        if self.direction == "uplink"
                        else self.num_tx_ports
                    ),
                    rank,
                ],
                dtype=self.cdtype,
            )
            # Store serving channel singular values for signal power [B, UT, F, R]
            s_srv_all = tf.zeros([B, N_UT_Total, FFT, rank], dtype=self.rdtype)

            # --- Pass 1: Determine Digital Beamforming Weights (SVD) ---
            for i in range(num_batches_ut_sinr):
                start_ut = i * batch_size_ut_sinr
                end_ut = tf.minimum(start_ut + batch_size_ut_sinr, N_UT_Total)
                srv_indices = self.neighbor_indices[:, start_ut:end_ut, 0:1]

                # Fetch only serving link in Port Domain
                h_srv_port = self.channel_interface.get_neighbor_channel_info(
                    B,
                    self.ut_loc[:, start_ut:end_ut, :],
                    self.bs_loc,
                    self.ut_orientations[:, start_ut:end_ut, :],
                    self.bs_orientations,
                    srv_indices,
                    self.ut_velocities[:, start_ut:end_ut, :],
                    self.in_state[:, start_ut:end_ut],
                    return_element_channel=False,
                    return_s_u_v=False,
                )

                # h_srv: [B, BUT, F, RxP, TxP]
                h_srv = tf.transpose(h_srv_port[:, :, 0, :, :, 0, :], [0, 1, 4, 2, 3])
                s_s, u_s, v_s = tf.linalg.svd(h_srv)

                # Reshape SVD results for scatter update (Remove B and BUT leading dims)
                # target shape for each update: [FFT, Port, Rank]
                u_s_update = tf.reshape(
                    u_s[..., :rank], [end_ut - start_ut, FFT, -1, rank]
                )
                v_s_update = tf.reshape(
                    v_s[..., :rank], [end_ut - start_ut, FFT, -1, rank]
                )
                s_s_update = tf.reshape(s_s[..., :rank], [end_ut - start_ut, FFT, rank])

                # Store Weights Mapped by UT/BS ID
                indices_ut = tf.stack(
                    [
                        tf.zeros([end_ut - start_ut], dtype=tf.int32),
                        tf.range(start_ut, end_ut),
                    ],
                    axis=-1,
                )
                bs_ids = tf.cast(srv_indices[:, :, 0], tf.int32)[
                    0
                ]  # Extract BUT BS IDs
                indices_bs = tf.stack(
                    [tf.zeros([end_ut - start_ut], dtype=tf.int32), bs_ids], axis=-1
                )

                if self.direction == "uplink":
                    w_ut_dig = tf.tensor_scatter_nd_update(
                        w_ut_dig, indices_ut, v_s_update
                    )
                    w_bs_dig = tf.tensor_scatter_nd_update(
                        w_bs_dig, indices_bs, u_s_update
                    )
                else:
                    w_ut_dig = tf.tensor_scatter_nd_update(
                        w_ut_dig, indices_ut, u_s_update
                    )
                    w_bs_dig = tf.tensor_scatter_nd_update(
                        w_bs_dig, indices_bs, v_s_update
                    )

                s_srv_all = tf.tensor_scatter_nd_update(
                    s_srv_all, indices_ut, s_s_update
                )

            # --- Pass 2: Interference Calculation (Port-Domain Loop) ---
            # Global Interference Buffer (for Uplink)
            interference_buffer_bs = tf.zeros([B, N_BS, FFT, rank], dtype=self.rdtype)

            for i in range(num_batches_ut_sinr):
                start_ut = i * batch_size_ut_sinr
                end_ut = tf.minimum(start_ut + batch_size_ut_sinr, N_UT_Total)
                current_batch_size = end_ut - start_ut

                # Get Neighbor channels in Port Domain
                sliced_neighbor_indices = self.neighbor_indices[:, start_ut:end_ut, :]
                h_batch_port = self.channel_interface.get_neighbor_channel_info(
                    B,
                    self.ut_loc[:, start_ut:end_ut, :],
                    self.bs_loc,
                    self.ut_orientations[:, start_ut:end_ut, :],
                    self.bs_orientations,
                    sliced_neighbor_indices,
                    self.ut_velocities[:, start_ut:end_ut, :],
                    self.in_state[:, start_ut:end_ut],
                    False,
                    False,
                )

                # h_batch_port: [B, BUT, K, RxP, TxP, 1, F]
                # Signal Power (Neighbor 0)
                p_alloc_batch = tf.reshape(
                    tf.gather(p_alloc, tf.range(start_ut, end_ut), axis=1),
                    [B, current_batch_size, 1, 1],
                )
                s_p = p_alloc_batch * tf.square(
                    tf.gather(s_srv_all, tf.range(start_ut, end_ut), axis=1)
                )
                s_power_all.append(s_p)

                # Interference (Neighbors 1...)
                neighbor_ids = sliced_neighbor_indices[:, :, 1:]
                h_int = tf.transpose(
                    h_batch_port[:, :, 1:, :, :, 0, :], [0, 1, 2, 5, 3, 4]
                )  # [B, BUT, K-1, F, RxP, TxP]

                if self.direction == "uplink":
                    # UL: Leakage to neighbor BSs
                    # v_self_dig: [B, BUT, F, TxP, Rank]
                    v_self = tf.gather(w_ut_dig, tf.range(start_ut, end_ut), axis=1)
                    # u_neighbor_dig: [B, BUT, K-1, F, RxP, Rank]
                    u_neighbor = tf.gather(
                        w_bs_dig, tf.cast(neighbor_ids, tf.int32), axis=1, batch_dims=1
                    )

                    # Heff_i = U_neighbor^H * H_int * V_self
                    hv = tf.einsum("bukfpt,buftx->bukfpx", h_int, v_self)
                    heff = tf.einsum(
                        "bukfpx,bukfpx->bukfx", tf.math.conj(u_neighbor), hv
                    )
                    p_leak = p_alloc_batch[:, :, :, None, None] * tf.square(
                        tf.abs(heff)
                    )  # [B, BUT, K-1, F, R]

                    # Accumulate to global buffer
                    indices = tf.stack(
                        [
                            tf.zeros(tf.shape(neighbor_ids), dtype=tf.int32),
                            tf.cast(neighbor_ids, tf.int32),
                        ],
                        axis=-1,
                    )
                    interference_buffer_bs = tf.tensor_scatter_nd_add(
                        interference_buffer_bs,
                        tf.reshape(indices, [-1, 2]),
                        tf.reshape(p_leak, [-1, FFT, rank]),
                    )
                else:
                    # DL: Interference from neighbor BSs
                    # u_self_dig: [B, BUT, F, RxP, Rank]
                    u_self = tf.gather(w_ut_dig, tf.range(start_ut, end_ut), axis=1)
                    # v_neighbor_dig: [B, BUT, K-1, F, TxP, Rank]
                    v_neighbor = tf.gather(
                        w_bs_dig, tf.cast(neighbor_ids, tf.int32), axis=1, batch_dims=1
                    )

                    # Heff_i = U_self^H * H_int * V_neighbor
                    hv = tf.einsum("bukfpt,bukftx->bukfpx", h_int, v_neighbor)
                    heff = tf.einsum("bufpx,bukfpx->bukfx", tf.math.conj(u_self), hv)
                    p_int = tf.expand_dims(p_alloc_batch, 2) * tf.square(tf.abs(heff))
                    i_total_all.append(tf.reduce_sum(p_int, axis=2))

            # --- Phase 2: Final SINR and Throughput Calculation ---
            s_power = tf.concat(s_power_all, axis=1)

            if self.direction == "uplink":
                # Fetch accumulated interference from buffer for each UE's serving BS
                i_total = tf.gather(
                    interference_buffer_bs,
                    tf.cast(self.serving_bs_ids, tf.int32),
                    axis=1,
                    batch_dims=1,
                )
            else:
                i_total = tf.concat(i_total_all, axis=1)

            # Final SINR
            noise_power = self.no  # Scalar or [SC]
            sinr = s_power / (i_total + noise_power)
            sinr_db = 10.0 * tf.math.log(tf.maximum(sinr, 1e-20)) / tf.math.log(10.0)

            # 4. Throughput Calculation (Simple Capacity for now or use MCS)
            throughput = tf.reduce_mean(
                tf.math.log(1.0 + sinr) / tf.math.log(2.0), axis=[2, 3]
            )
            # [B, U]

            # --- End New SINR Logic ---

            # Expand dimensions to match legacy shape [Batch, UT, OFDM(1), SC, Streams]
            sinr = tf.expand_dims(sinr, axis=2)  # Add OFDM dim

            # For logging, we might need other metrics, but SINR is key.
            sinr_db = 10.0 * tf.math.log(tf.maximum(sinr, 1e-20)) / tf.math.log(10.0)
            sinr_eff_avg = tf.reduce_mean(sinr, axis=[-1, -2, -3])

            # Recalculate rank_per_user for logging (fixed at 1 for now)
            num_streams = rank
            rank_per_user = tf.fill(
                [self.batch_size, self.num_ut], tf.cast(num_streams, tf.float32)
            )

            # Use Discrete MCS Table Lookup
            # Returns Spectral Efficiency (bits/symbol) including BLER penalty
            capacity_per_re, mcs_idx = self.mcs_adapter.get_throughput_vectorized(
                sinr_db
            )

            # If using RBG granularity, each point represents rbg_size_sc subcarriers
            if self.config.use_rbg_granularity:
                # Scale capacity by the size of the RBG
                rbg_scale = tf.cast(self.rbg_size_sc, self.rdtype)
                capacity_per_re = capacity_per_re * rbg_scale

            throughput_per_user = tf.reduce_sum(capacity_per_re, axis=[-1, -2])

            # --- Record Results ---
            # Prepare metrics for recording
            # Reshape/Cast as needed to match get_hist expectations
            # history keys: [batch, num_bs, num_ut_per_sector]
            # Current variables are [batch, num_ut] where num_ut = num_bs * num_ut_per_sector
            # We need to reshape [batch, num_ut] -> [batch, num_bs, num_ut_per_sector]

            def match_hist_shape(tensor):
                # tensor: [batch, num_ut, ...]
                # Reduce extra dimensions (ofdm, sc, streams) by averaging
                rank = tensor.shape.rank
                if rank is not None and rank > 2:
                    tensor = tf.reduce_mean(tensor, axis=list(range(2, rank)))

                return tf.reshape(
                    tensor, [self.batch_size, self.num_bs, self.num_ut_per_sector]
                )

            # Average MCS index over streams/subcarriers
            mcs_idx_avg = tf.reduce_mean(tf.cast(mcs_idx, tf.float32), axis=[-1, -2])

            # Record using 'drop_idx' as the time index
            hist = record_results(
                hist,
                drop_idx,
                sim_failed=False,
                pathloss_serving_cell=match_hist_shape(pl_db),
                num_allocated_re=match_hist_shape(
                    tf.fill(
                        [self.batch_size, self.num_ut],
                        float(self.resource_grid.num_effective_subcarriers),
                    )
                ),  # Placeholder
                tx_power_per_ut=match_hist_shape(total_power),
                num_decoded_bits=match_hist_shape(throughput_per_user),
                mcs_index=match_hist_shape(mcs_idx_avg),
                harq_feedback=match_hist_shape(
                    tf.zeros([self.batch_size, self.num_ut])
                ),
                olla_offset=match_hist_shape(tf.zeros([self.batch_size, self.num_ut])),
                sinr_eff=match_hist_shape(sinr_eff_avg),
                p_cmax_dbm=match_hist_shape(p_cmax_dbm_tensor),
                rank=match_hist_shape(rank_per_user),
                mpr_db=match_hist_shape(mpr_db),
                pf_metric=tf.reshape(
                    match_hist_shape(tf.zeros([self.batch_size, self.num_ut])),
                    [self.batch_size, self.num_bs, 1, 1, self.num_ut_per_sector],
                ),  # Reshaped for get_hist.py's axis=[-2, -3] reduction
            )

            # No time evolution; topology is reset in next iteration

        # Stack history to convert TensorArrays to Tensors
        final_hist = {}
        for key in hist:
            if isinstance(hist[key], tf.TensorArray):
                final_hist[key] = hist[key].stack()
            else:
                final_hist[key] = hist[key]

        return final_hist
