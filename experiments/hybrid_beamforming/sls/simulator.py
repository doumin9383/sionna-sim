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


from .my_configs import HybridSLSConfig


class HybridSystemSimulator(Block):

    def __init__(
        self,
        config: HybridSLSConfig,
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

        # Instantiate ResourceGrid from config (Factory)
        self.resource_grid = ResourceGrid(
            # num_ofdm_symbols=config.num_ofdm_symbols,
            num_ofdm_symbols=1,  # チャネル生成のためだけに使うので1でOK
            fft_size=config.num_subcarriers,
            subcarrier_spacing=config.subcarrier_spacing,
            # pilot_pattern=rg_config.pilot_pattern,
            # pilot_ofdm_symbol_indices=rg_config.pilot_ofdm_symbol_indices,
        )

        # Instantiate Antenna Arrays from config
        self.bs_array = config.bs_array
        self.ut_array = config.ut_array

        self.num_ut_ant = self.ut_array.num_ant
        self.num_bs_ant = self.bs_array.num_ant

        if self.direction == "uplink":
            self.num_tx, self.num_rx = self.num_ut, self.num_bs
            self.num_tx_ant, self.num_rx_ant = self.num_ut_ant, self.num_bs_ant
            self.num_tx_per_sector = self.num_ut_per_sector
        else:
            self.num_tx, self.num_rx = self.num_bs, self.num_ut
            self.num_tx_ant, self.num_rx_ant = self.num_bs_ant, self.num_ut_ant
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
        self._setup_channel_model(
            config.scenario,
            config.carrier_frequency,
            o2i_model,
            self.ut_array,
            self.bs_array,
            average_street_width,
            average_building_height,
        )

        # Generate multicell topology
        # Moved to call method loop for drops
        # self._setup_topology(config.num_rings, min_bs_ut_dist, max_bs_ut_dist)

        if self.direction == "uplink":
            num_tx_ports = config.ut_num_rf_chains
            num_rx_ports = config.bs_num_rf_chains
        else:
            num_tx_ports = config.bs_num_rf_chains
            num_rx_ports = config.ut_num_rf_chains

        if self.direction == "uplink":
            self.sim_tx_array = self.ut_array
            self.sim_rx_array = self.bs_array
        else:
            self.sim_tx_array = self.bs_array
            self.sim_rx_array = self.ut_array

        # Instantiate the Hybrid Channel Interface
        self.channel_interface = HybridChannelInterface(
            channel_model=self.channel_model,
            resource_grid=self.resource_grid,
            tx_array=self.sim_tx_array,
            rx_array=self.sim_rx_array,
            num_tx_ports=num_tx_ports,
            num_rx_ports=num_rx_ports,
            precision=self.precision,
            use_rbg_granularity=config.use_rbg_granularity,
            rbg_size_sc=self.rbg_size_sc if self.rbg_size_sc else 1,
            neighbor_indices=None,  # Topology is set in the loop
            external_loader=self.external_loader,
        )

        # Instantiate simplified link adaptation (Physics Abstraction for SINR)
        # self.phy_abstraction was removed. We calculate SINR directly in call().

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

        if self.config.topology_type == "HexGrid":
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
        else:
            raise NotImplementedError(
                f"Topology type {self.config.topology_type} not implemented in _setup_topology"
            )

        # 1. Calculate distances [batch, num_ut, num_bs]
        # ut_loc: [batch, num_ut, 3], bs_loc: [batch, num_bs, 3]
        diff = tf.expand_dims(self.ut_loc, axis=2) - tf.expand_dims(self.bs_loc, axis=1)
        dist = tf.norm(diff, axis=-1)

        # 2. Select top K neighbors based on distance
        # Note: smallest distance = largest -dist
        _, neighbor_indices = tf.math.top_k(-dist, k=self.config.num_neighbors)
        # neighbor_indices shape: [batch, num_ut, num_neighbors]
        self.neighbor_indices = neighbor_indices

        # 3. Set topology in channel model
        if self.external_loader is None:
            self.channel_model.set_topology(
                self.ut_loc,
                self.bs_loc,
                self.ut_orientations,
                self.bs_orientations,
                self.ut_velocities,
                self.in_state,
                self.los,
                self.bs_virtual_loc,
            )
        else:
            # When using external data, we still need mesh indices
            self.ut_mesh_indices = self.external_loader.find_nearest_mesh(self.ut_loc)

    # @tf.function(jit_compile=False)
    def call(self, num_drops, tx_power_dbm):
        # Initialize result history
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
            # This generates new UT locations, channels, etc.
            self._setup_topology(
                self.config.num_rings,
                self.config.min_bs_ut_dist,
                self.config.max_bs_ut_dist,
            )

            # 1. Get Channel Information
            # Get Element-Domain Channel for Beam Selection
            h_elem_bs = self.channel_interface.get_element_channel_for_beam_selection(
                batch_size=self.batch_size,
                rbg_size_sc=self.rbg_size_sc,
                ut_loc=self.ut_loc,
                bs_loc=self.bs_loc,
                ut_orient=self.ut_orientations,
                bs_orient=self.bs_orientations,
                neighbor_indices=self.neighbor_indices,
                ut_velocities=self.ut_velocities,
                in_state=self.in_state,
            )[
                0
            ]  # get_element returns (h, s, u, v) from SVD call inside? No, I defined it to return h_elem?
            # Wait, get_element_channel_for_beam_selection implementation:
            # It returns "return h, s, u, v" because it copies logic from get_full_channel_info/calls svd?
            # In my previous edit Step 57:
            # s, u, v = tf.linalg.svd(h)
            # return h, s, u, v
            # So I should take index [0].

            # 2. Select Analog Beams (BS Side)
            # BS Beam Selection
            # 2. Select Analog Beams (BS Side)
            # BS Beam Selection
            # w_rf_selected: [Batch, U, TotalAnt, SubPorts(2)]
            w_rf_selected = self.beam_selector(h_elem_bs)

            # Reshape to [Batch, TotalAnt, U * SubPorts] -> Mapping Users/Pols to RF Chains
            # Transpose to [Batch, TotalAnt, U, SubPorts]
            w_rf_transposed = tf.transpose(w_rf_selected, perm=[0, 2, 1, 3])
            # Reshape to [Batch, TotalAnt, U*SubPorts]
            # w_rf_flat: [Batch, TotalAnt, Candidates]
            w_rf_flat = tf.reshape(
                w_rf_transposed, [self.batch_size, self.num_bs_ant, -1]
            )

            # Match number of RF chains (BS Ports)
            if self.direction == "uplink":
                target_ports = self.channel_interface.hybrid_channel.num_rx_ports
            else:
                target_ports = self.channel_interface.hybrid_channel.num_tx_ports

            num_candidates = tf.shape(w_rf_flat)[-1]

            if num_candidates >= target_ports:
                w_rf_bs = w_rf_flat[..., :target_ports]
            else:
                padding = tf.zeros(
                    [self.batch_size, self.num_bs_ant, target_ports - num_candidates],
                    dtype=w_rf_flat.dtype,
                )
                w_rf_bs = tf.concat([w_rf_flat, padding], axis=-1)

            # UE Analog Beam: Identity (Full Digital or Fixed)
            if self.direction == "uplink":
                ue_ports = self.channel_interface.hybrid_channel.num_tx_ports
            else:
                ue_ports = self.channel_interface.hybrid_channel.num_rx_ports

            a_rf_ue = tf.eye(self.num_ut_ant, num_columns=ue_ports, dtype=tf.complex64)

            # Apply weights
            if self.direction == "uplink":
                self.channel_interface.set_analog_weights(w_rf=a_rf_ue, a_rf=w_rf_bs)
            else:
                self.channel_interface.set_analog_weights(w_rf=w_rf_bs, a_rf=a_rf_ue)

            # 3. Get Precoding Channel (Effective Channel)
            # Get Full Channel for Interference/SINR Calc
            h, s_all, u_all, v_all = self.channel_interface.get_full_channel_info(
                self.batch_size,
                ut_loc=self.ut_loc,
                bs_loc=self.bs_loc,
                ut_orient=self.ut_orientations,
                bs_orient=self.bs_orientations,
                neighbor_indices=self.neighbor_indices,
                ut_velocities=self.ut_velocities,
                in_state=self.in_state,
            )

            # Compute SVD on Coarse Channel
            s_prec, u_prec, v_prec = tf.linalg.svd(h_prec)

            # Expand v_prec to Full Bandwidth
            # v_prec: [batch, num_ut, num_bs, ofdm, num_blocks, tx_ports, tx_ports]
            # We need to expand dim -3 (num_blocks) to num_sc

            # Determine effective target dimensions
            if self.config.use_rbg_granularity:
                # RBGモードの場合、h の周波数軸はすでに RBG 数になっている
                eff_total_subcarriers = h.shape[4]
                eff_rbg_size_sc = 1
            else:
                # RBモード（RB単位）
                eff_total_subcarriers = self.resource_grid.num_effective_subcarriers
                eff_rbg_size_sc = self.rbg_size_sc

            # v_prec: [batch, num_ut, num_bs, ofdm, num_blocks, tx_ports, tx_ports]
            # ユーザー指摘によりRank 1制限を撤廃し、全ストリーム(レイヤー)を使用する
            # v_prec は SVD の結果 [..., tx, k] (k=min(tx,rx))

            v_expanded = expand_precoder(
                v_prec,
                total_subcarriers=eff_total_subcarriers,
                granularity_type=self.precoding_granularity,
                rbg_size_sc=eff_rbg_size_sc,
            )

            # 3. Serving Link 抽出
            # The serving link is always at index 0 of the neighbor list by definition of top_k
            serving_link_idx = tf.zeros([self.batch_size, self.num_ut], dtype=tf.int32)

            # u_prec: [batch, num_ut, num_bs, ofdm, num_blocks, rx_ports, rx_streams (min(rx,tx))]
            u_expanded = expand_precoder(
                u_prec,
                total_subcarriers=eff_total_subcarriers,
                granularity_type=self.precoding_granularity,
                rbg_size_sc=eff_rbg_size_sc,
            )

            # 3. Serving Link 抽出
            # The serving link is always at index 0 of the neighbor list by definition of top_k
            serving_link_idx = tf.zeros([self.batch_size, self.num_ut], dtype=tf.int32)

            u_serv = tf.gather(u_expanded, serving_link_idx, axis=2, batch_dims=2)
            # v_serv calculation is not strictly needed for interference, but v_expanded contains it.

            # 4. 干渉計算 (Downlink 想定の効率化)
            # h: [batch, ut, bs_neighbor, ofdm, sc, rx_ant, tx_ant]
            # v_expanded (neighbor_precoders): [batch, ut, bs_neighbor, ofdm, sc, tx_ant, 1]

            # UT i の受信ビームフォーマーをチャネルに適用
            # h_u: [batch, ut, bs_neighbor, ofdm, sc, 1, tx_ant]
            h_u = tf.einsum("buosrp,bujosrt->bujospt", tf.math.conj(u_serv), h)

            # 他局のプリコーダーとの結合
            # h_eff: [batch, ut, bs_neighbor, ofdm, sc, 1, 1]
            h_eff = tf.einsum("bujospt,bujostq->bujospq", h_u, v_expanded)

            # パワー計算
            # h_eff: [batch, ut, bs_neighbor, ofdm, sc, rx_streams, tx_streams]
            # 電力そのもの: [batch, ut, bs_neighbor, ofdm, sc, rx_streams, tx_streams]
            power_tensor = tf.square(tf.abs(h_eff))

            # -------------------------------------------------------
            # 信号成分 (Serving Link: Neighbor index 0)
            # -------------------------------------------------------
            # Serving Link の h_eff は、理想的には対角行列（SVDによる直交化）に近い
            # 信号成分 = 対角成分 |h_ii|^2
            # power_tensor[:, :, 0, ...] -> [batch, ut, ofdm, sc, rx_streams, tx_streams]
            serving_power_matrix = power_tensor[:, :, 0, :, :, :, :]
            s_power = tf.linalg.diag_part(
                serving_power_matrix
            )  # [batch, ut, ofdm, sc, streams]

            # -------------------------------------------------------
            # 干渉成分 (Interfering Links: Neighbor index > 0)
            # -------------------------------------------------------
            # 干渉リンクからは、全送信ストリームの電力が雑音として加算される
            # neighbor index 1以降を合計
            # interference_power_matrix: [batch, ut, neighbor-1, ofdm, sc, rx_streams, tx_streams]
            interference_power_matrix = power_tensor[:, :, 1:, :, :, :, :]

            # Reduce sum over neighbors (axis 2) and tx_streams (axis -1)
            # Result: [batch, ut, ofdm, sc, rx_streams]
            interference_total = tf.reduce_sum(interference_power_matrix, axis=[2, -1])

            # Effective Noise per stream: N0 + Interference
            noise_plus_interference = self.no + interference_total

            # For Power Control: Identify Serving BS
            serving_bs_idx = self.neighbor_indices[:, :, 0]
            serving_bs_idx_i32 = tf.cast(serving_bs_idx, tf.int32)

            # 4. Power Control & Link Adaptation
            # a. Calculate Path Loss (Simple Euclidean distance based approximation for PC)
            # Find serving BS location
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

            # c. Calculate Tx Power
            if self.direction == "uplink":
                # num_rbs: Total RBs (assuming full bw allocation for now or partial)
                # resource_grid.num_effective_subcarriers / 12
                num_rbs = self.resource_grid.num_effective_subcarriers / 12.0
                p_tx_dbm = self.power_control.calculate_tx_power(
                    pl_db, num_rbs, mpr_val
                )
            else:
                # Downlink: Use fixed power
                p_tx_dbm = tx_power_dbm

            # Broadcast p_tx_dbm to [batch, num_ut] if it calculated scalar/vector
            # p_tx_dbm might be tensor [batch, num_ut]
            total_power = dbm_to_watt(p_tx_dbm)

            # Reshape/Broadcast for broadcasting: [batch, num_ut, 1, 1, 1]
            if len(total_power.shape) == 0:  # Scalar
                total_power = tf.broadcast_to(
                    total_power, [self.batch_size, self.num_ut]
                )

            # Ensure shape is [batch, num_ut, 1, 1, 1] for waterfilling
            total_power_expanded = tf.reshape(
                total_power, [self.batch_size, self.num_ut, 1, 1, 1]
            )

            # d. Physics Abstraction (Equal Power Allocation across streams -> SINR)
            # num_streams is dynamic based on channel rank (SVD)
            num_streams = tf.shape(s_power)[-1]
            p_alloc = total_power_expanded / tf.cast(num_streams, self.rdtype)

            # Calculate SINR: P * |h|^2 / (N0 + I)
            # s_power is signal power (equivalent to |h|^2)
            # sinr: [batch, ut, ofdm, sc, streams]
            sinr = (p_alloc * s_power) / noise_plus_interference

            # e. MCS Selection & Throughput
            # MCS Adapter expects SINR in dB
            sinr_db = 10.0 * tf.math.log(tf.maximum(sinr, 1e-20)) / tf.math.log(10.0)

            # Effective SINR for reporting
            # Average over subcarriers and streams for logging, but use per-stream for throughput
            sinr_eff_avg = tf.reduce_mean(
                sinr, axis=[-1, -2, -3]
            )  # avg over ofdm, sc, streams

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
