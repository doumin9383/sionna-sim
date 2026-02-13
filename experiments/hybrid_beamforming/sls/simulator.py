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
from ..shared import weight_utils

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

    def _setup_drop_topology(self, drop_idx):
        """シミュレーションの1ドロップ分のトポロジーセットアップ"""
        if self.external_loader is not None:
            self.external_loader.load_drop(drop_idx)

        # 新しいUT位置、チャネル等の生成
        self._setup_topology(
            self.config.num_rings,
            self.config.min_bs_ut_dist,
            self.config.max_bs_ut_dist,
        )

    def _select_analog_beams(self):
        """コードブックベースのアナログビーム選択を実行し、重みをセットする"""
        h_serv_list = []
        batch_size_beam = 1  # 1 UTずつ処理（メモリ節約）

        for i in range(0, self.num_ut, batch_size_beam):
            end_i = min(i + batch_size_beam, self.num_ut)
            curr_neigh_inds = self.neighbor_indices[:, i:end_i, :]
            curr_ut_loc = self.ut_loc[:, i:end_i, :]
            curr_ut_orient = self.ut_orientations[:, i:end_i, :]
            curr_ut_vel = self.ut_velocities[:, i:end_i, :]
            curr_in_state = self.in_state[:, i:end_i]

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
            # Neighbor=0, Time=0
            h_serv_batch = h_elem_batch[:, :, 0, :, :, 0, :]

            # 周波数方向のサブサンプリング（メモリ節約）
            if self.config.use_rbg_granularity:
                # RBG単位での周波数サンプリング（中心付近の1点を使用）
                # 完全に平均をとるのではなく、計算負荷を減らすために間引く
                h_serv_batch = h_serv_batch[..., :: self.rbg_size_sc]

            h_serv_list.append(h_serv_batch)

        h_serv = tf.concat(h_serv_list, axis=1)

        # BS側のビーム選択
        B = self.batch_size
        N_BS = self.num_bs
        N_UT_Sec = self.num_ut_per_sector
        N_UT_Total = self.num_ut
        RxA = tf.shape(h_serv)[2]
        TxA = tf.shape(h_serv)[3]
        FFT = tf.shape(h_serv)[4]

        batch_indices = tf.broadcast_to(tf.range(B)[:, None], [B, N_UT_Total])
        bs_indices = tf.cast(self.serving_bs_ids, tf.int32)
        indices = tf.stack(
            [batch_indices, bs_indices, tf.zeros_like(bs_indices)], axis=-1
        )

        h_serv_flat = tf.reshape(h_serv, [B, N_UT_Total, -1])
        h_bs_shape = [B, N_BS, N_UT_Sec, RxA * TxA * FFT]
        h_bs_flat = tf.scatter_nd(indices, h_serv_flat, h_bs_shape)
        h_bs = tf.reshape(h_bs_flat, [B, N_BS, N_UT_Sec, RxA, TxA, FFT])

        if self.direction == "uplink":
            h_bs_permuted = tf.transpose(h_bs, [0, 1, 2, 4, 3, 5])
        else:
            h_bs_permuted = tf.transpose(h_bs, [0, 1, 2, 3, 4, 5])

        h_selector_input = tf.reshape(
            h_bs_permuted,
            [-1, N_UT_Sec, tf.shape(h_bs_permuted)[3], self.num_bs_ant, FFT],
        )
        w_rf_bs_flat = self.beam_selector(h_selector_input, self.config.bs_array)
        w_rf_bs = tf.reshape(w_rf_bs_flat, [B, N_BS, self.num_bs_ant, -1])

        # UE側のアナログビーム（Identity）
        ue_ports = (
            self.num_tx_ports if self.direction == "uplink" else self.num_rx_ports
        )
        a_rf_ue = tf.eye(self.num_ut_ant, num_columns=ue_ports, dtype=tf.complex64)

        if self.direction == "uplink":
            self.channel_interface.set_analog_weights(w_rf=a_rf_ue, a_rf=w_rf_bs)
        else:
            self.channel_interface.set_analog_weights(w_rf=w_rf_bs, a_rf=a_rf_ue)

        return w_rf_bs

    def _compute_digital_weights(self, granularity="subband"):
        """SVDベースのデジタル重みを粒度（granularity）に応じて計算する"""
        B = self.batch_size
        N_BS = self.num_bs
        N_UT_Total = self.num_ut
        # Simulation Resolution (Ntarget) alignment
        if self.config.use_rbg_granularity:
            N_target = self.resource_grid.fft_size // self.rbg_size_sc
        else:
            N_target = self.resource_grid.num_effective_subcarriers

        rank = self.config.num_layers

        # UTバッチ処理用のバッファ
        batch_size_ut = self.config.batch_size_ut

        # デジタル重みのバッファ
        # UT側 (Terminal)
        w_ut_dig_full = tf.zeros(
            [
                B,
                N_UT_Total,
                N_target,
                (
                    self.num_tx_ports
                    if self.direction == "uplink"
                    else self.num_rx_ports
                ),
                rank,
            ],
            dtype=self.cdtype,
        )
        # BS側 (Sector)
        w_bs_dig_full = tf.zeros(
            [
                B,
                N_BS,
                N_target,
                (
                    self.num_rx_ports
                    if self.direction == "uplink"
                    else self.num_tx_ports
                ),
                rank,
            ],
            dtype=self.cdtype,
        )
        # 特異値（Serving Channel）
        s_srv_full = tf.zeros([B, N_UT_Total, N_target, rank], dtype=self.rdtype)

        for start_ut in range(0, N_UT_Total, batch_size_ut):
            end_ut = min(start_ut + batch_size_ut, N_UT_Total)
            curr_batch_size = end_ut - start_ut
            srv_indices = self.neighbor_indices[:, start_ut:end_ut, 0:1]

            # 1. 接続先BSとのポートドメインチャネルを取得
            h_srv_port = self.channel_interface.get_neighbor_channel_info(
                batch_size=B,
                ut_loc=self.ut_loc[:, start_ut:end_ut, :],
                bs_loc=self.bs_loc,
                ut_orient=self.ut_orientations[:, start_ut:end_ut, :],
                bs_orient=self.bs_orientations,
                neighbor_indices=srv_indices,
                ut_velocities=self.ut_velocities[:, start_ut:end_ut, :],
                in_state=self.in_state[:, start_ut:end_ut],
                return_element_channel=False,
                return_s_u_v=False,
            )
            # h_srv_port: [B, BUT, Neighbor(1), RxP, TxP, Time(1), F]
            # [B, BUT, F, RxP, TxP] に変換
            h_srv = tf.transpose(h_srv_port[:, :, 0, :, :, 0, :], [0, 1, 4, 2, 3])

            # 2. 粒度に応じた重み計算（SVD）
            s, u, v = weight_utils.compute_digital_weights(
                h_srv,
                granularity=granularity,
                rbg_size_sc=self.rbg_size_sc,
                weight_type="svd",
            )

            # 3. ターゲット解像度に展開
            # SVDの結果から必要なランク分をスライス
            s_exp = weight_utils.expand_weights(
                s[..., :rank], N_target, granularity, self.rbg_size_sc
            )
            u_exp = weight_utils.expand_weights(
                u[..., :rank], N_target, granularity, self.rbg_size_sc
            )
            v_exp = weight_utils.expand_weights(
                v[..., :rank], N_target, granularity, self.rbg_size_sc
            )

            # 4. バッファに格納
            # Batch方向も考慮したインデックス作成
            batch_indices = tf.range(B)[:, None]
            batch_indices = tf.broadcast_to(batch_indices, [B, curr_batch_size])
            ut_indices_range = tf.range(start_ut, end_ut)[None, :]
            ut_indices_range = tf.broadcast_to(ut_indices_range, [B, curr_batch_size])

            indices_ut = tf.stack([batch_indices, ut_indices_range], axis=-1)
            indices_ut_flat = tf.reshape(indices_ut, [-1, 2])

            bs_ids = tf.cast(srv_indices[:, :, 0], tf.int32)
            bs_ids_flat = tf.reshape(bs_ids, [B, curr_batch_size])

            indices_bs = tf.stack([batch_indices, bs_ids_flat], axis=-1)
            indices_bs_flat = tf.reshape(indices_bs, [-1, 2])

            # updates もフラットにする [B*BUT, F, P, Rank]
            u_exp_flat = tf.reshape(u_exp, [-1, N_target, tf.shape(u_exp)[-2], rank])
            v_exp_flat = tf.reshape(v_exp, [-1, N_target, tf.shape(v_exp)[-2], rank])
            s_exp_flat = tf.reshape(s_exp, [-1, N_target, rank])

            if self.direction == "uplink":
                w_ut_dig_full = tf.tensor_scatter_nd_update(
                    w_ut_dig_full, indices_ut_flat, v_exp_flat
                )
                w_bs_dig_full = tf.tensor_scatter_nd_update(
                    w_bs_dig_full, indices_bs_flat, u_exp_flat
                )
            else:
                w_ut_dig_full = tf.tensor_scatter_nd_update(
                    w_ut_dig_full, indices_ut_flat, u_exp_flat
                )
                w_bs_dig_full = tf.tensor_scatter_nd_update(
                    w_bs_dig_full, indices_bs_flat, v_exp_flat
                )

            s_srv_full = tf.tensor_scatter_nd_update(
                s_srv_full, indices_ut_flat, s_exp_flat
            )

        return w_ut_dig_full, w_bs_dig_full, s_srv_full

    def _apply_power_control(self, tx_power_dbm):
        """パスロスと最大電力制約を考慮した送信電力を計算する"""
        B = self.batch_size
        N_UT = self.num_ut

        # 1. 接続先BSの特定
        serving_bs_idx = self.neighbor_indices[:, :, 0]
        serving_bs_idx_i32 = tf.cast(serving_bs_idx, tf.int32)
        serving_bs_loc = tf.gather(
            self.bs_loc, serving_bs_idx_i32, axis=1, batch_dims=1
        )
        dist = tf.norm(self.ut_loc - serving_bs_loc, axis=-1)

        # 2. パスロス計算
        if self.external_loader is not None:
            powers_dbm = self.external_loader.get_power_map(self.ut_mesh_indices)
            serving_bs_idx_expand = tf.expand_dims(serving_bs_idx_i32, axis=-1)
            serving_power = tf.gather(
                powers_dbm, serving_bs_idx_expand, axis=2, batch_dims=2
            )
            serving_power = tf.squeeze(serving_power, axis=-1)
            pl_db = self.config.bs_max_power_dbm - serving_power
        else:
            fc_ghz = self.config.carrier_frequency / 1e9
            dist_safe = tf.maximum(dist, 1.0)
            pl_db = (
                28.0
                + 22.0 * tf.math.log(dist_safe) / tf.math.log(10.0)
                + 20.0 * tf.math.log(fc_ghz) / tf.math.log(10.0)
            )

        # 3. MPR (Maximum Power Reduction)
        mpr_val = self.mpr_model.get_mpr("CP-OFDM", self.config.num_layers)
        mpr_db = tf.fill([B, N_UT], tf.cast(mpr_val, tf.float32))

        # 4. Tx Power 計算
        if self.direction == "uplink":
            num_rbs = self.resource_grid.num_effective_subcarriers / 12.0
            p_tx_dbm = self.power_control.calculate_tx_power(pl_db, num_rbs, mpr_val)
            p_cmax_dbm = self.ut_max_power_dbm - mpr_val
        else:
            p_tx_dbm = tx_power_dbm
            p_cmax_dbm = self.bs_max_power_dbm

        p_cmax_dbm_tensor = tf.fill([B, N_UT], tf.cast(p_cmax_dbm, tf.float32))
        p_tx_watt = dbm_to_watt(p_tx_dbm)
        if len(p_tx_watt.shape) == 0:
            p_tx_watt = tf.broadcast_to(p_tx_watt, [B, N_UT])

        return p_tx_watt, pl_db, mpr_db, p_cmax_dbm_tensor

    def _process_sinr_and_la(self, w_ut_dig, w_bs_dig, s_srv, p_tx_watt):
        """SINR計算、干渉計算、Link Adaptationを実行する"""
        B = self.batch_size
        N_BS = self.num_bs
        N_UT = self.num_ut
        if self.config.use_rbg_granularity:
            N_target = self.resource_grid.fft_size // self.rbg_size_sc
        else:
            N_target = self.resource_grid.num_effective_subcarriers

        rank = self.config.num_layers

        # 1. 電力割り当て: レイヤーごとに均等分配
        p_layer = p_tx_watt / tf.cast(rank, self.rdtype)
        p_layer_expanded = tf.reshape(p_layer, [B, N_UT, 1, 1])

        s_power_all = []
        i_total_all = []
        # [B, N_BS, N_target, rank]
        interference_buffer_bs = tf.zeros([B, N_BS, N_target, rank], dtype=self.rdtype)

        batch_size_ut = self.config.batch_size_ut
        for start_ut in range(0, N_UT, batch_size_ut):
            end_ut = min(start_ut + batch_size_ut, N_UT)
            curr_batch_size = end_ut - start_ut
            sliced_neigh_inds = self.neighbor_indices[:, start_ut:end_ut, :]

            # 隣接チャネル（ポートドメイン）取得
            # h_batch_port: [B, BUT, Neighbor, RxP, TxP, Time(1), N_target]
            h_batch_port = self.channel_interface.get_neighbor_channel_info(
                B,
                self.ut_loc[:, start_ut:end_ut, :],
                self.bs_loc,
                self.ut_orientations[:, start_ut:end_ut, :],
                self.bs_orientations,
                sliced_neigh_inds,
                self.ut_velocities[:, start_ut:end_ut, :],
                self.in_state[:, start_ut:end_ut],
                False,
                False,
            )

            # 希望信号電力 (Serving link is neighbor 0)
            # s_srv: [B, N_UT, N_target, rank]
            # p_layer_expanded: [B, N_UT, 1, 1] -> [B, BUT, 1, rank]
            s_p = p_layer_expanded[:, start_ut:end_ut] * tf.square(
                s_srv[:, start_ut:end_ut]
            )
            s_power_all.append(s_p)

            # 干渉計算 (Neighbor 1+)
            neighbor_ids = sliced_neigh_inds[:, :, 1:]
            # [B, BUT, K-1, f, RxP, TxP]
            h_int = tf.transpose(h_batch_port[:, :, 1:, :, :, 0, :], [0, 1, 2, 5, 3, 4])

            if self.direction == "uplink":
                # v_self: [B, BUT, f, TxP, rank]
                v_self = w_ut_dig[:, start_ut:end_ut]
                # u_neighbor: [B, BUT, K-1, f, RxP, rank]
                u_neighbor = tf.gather(
                    w_bs_dig, tf.cast(neighbor_ids, tf.int32), axis=1, batch_dims=1
                )

                # Heff = U^H * H * V
                # hv = H * V -> [B, BUT, K-1, f, RxP, rank_tx]
                hv = tf.einsum("bukfpt,buftx->bukfpx", h_int, v_self)
                # heff = U^H * hv -> [B, BUT, K-1, f, rank_rx, rank_tx]
                heff = tf.einsum("bukfpx,bukfpx->bukfx", tf.math.conj(u_neighbor), hv)
                # Wait, the above einsum reduces RX axis but also Rank axis?
                # heff should be per-layer.
                # heff_full = conj(u_neighbor) [B, BUT, K-1, f, RxP, rank_rx] * hv [B, BUT, K-1, f, RxP, rank_tx]
                # -> [B, BUT, K-1, f, rank_rx, rank_tx]
                heff_matrix = tf.einsum(
                    "bukfpr,bukfpt->bukfrt", tf.math.conj(u_neighbor), hv
                )

                # p_leak: layer l of neighbor BS is affected by all layers of this UE
                # p_leak_l = sum_m |heff_l,m|^2 * p_layer
                p_leak = p_layer_expanded[
                    :, start_ut:end_ut, :, None, None
                ] * tf.square(tf.abs(heff_matrix))
                p_leak_per_layer = tf.reduce_sum(
                    p_leak, axis=-1
                )  # [B, BUT, K-1, f, rank_rx]

                # 干渉の蓄積
                batch_indices = tf.range(B)[:, None, None]
                batch_indices = tf.broadcast_to(batch_indices, tf.shape(neighbor_ids))
                indices = tf.stack(
                    [batch_indices, tf.cast(neighbor_ids, tf.int32)], axis=-1
                )

                interference_buffer_bs = tf.tensor_scatter_nd_add(
                    interference_buffer_bs,
                    tf.reshape(indices, [-1, 2]),
                    tf.reshape(p_leak_per_layer, [-1, N_target, rank]),
                )
            else:
                # u_self: [B, BUT, f, RxP, rank]
                u_self = w_ut_dig[:, start_ut:end_ut]
                # v_neighbor: [B, BUT, K-1, f, TxP, rank]
                v_neighbor = tf.gather(
                    w_bs_dig, tf.cast(neighbor_ids, tf.int32), axis=1, batch_dims=1
                )

                # hv = H * V -> [B, BUT, K-1, f, RxP, rank_tx]
                hv = tf.einsum("bukfpt,bukftx->bukfpx", h_int, v_neighbor)
                # heff_matrix: [B, BUT, K-1, f, rank_rx, rank_tx]
                heff_matrix = tf.einsum(
                    "bufpr,bukfpx->bukfrx", tf.math.conj(u_self), hv
                )

                # p_int_l = sum_m |heff_l,m|^2 * p_layer
                p_int = p_layer_expanded[:, start_ut:end_ut, :, None, None] * tf.square(
                    tf.abs(heff_matrix)
                )
                p_int_per_layer = tf.reduce_sum(
                    p_int, axis=-1
                )  # [B, BUT, K-1, f, rank_rx]

                # UT側での干渉合算: 全ての近隣BSからの干渉を合算
                i_total_all.append(
                    tf.reduce_sum(p_int_per_layer, axis=2)
                )  # [B, BUT, f, rank]

        # 最終SINR
        s_power = tf.concat(s_power_all, axis=1)  # [B, N_UT, N_target, rank]
        if self.direction == "uplink":
            i_total = tf.gather(
                interference_buffer_bs,
                tf.cast(self.serving_bs_ids, tf.int32),
                axis=1,
                batch_dims=1,
            )  # [B, N_UT, N_target, rank]
        else:
            i_total = tf.concat(i_total_all, axis=1)  # [B, N_UT, N_target, rank]

        sinr = s_power / (i_total + self.no)
        sinr_db = 10.0 * tf.math.log(tf.maximum(sinr, 1e-20)) / tf.math.log(10.0)

        # Link Adaptation
        # throughput_vectorized は [sinr] を受け取りスペクトル効率とMCSを返す
        # sinr: [B, N_UT, N_target, rank]
        capacity_per_re, mcs_idx = self.mcs_adapter.get_throughput_vectorized(sinr_db)

        # RBG単位での計算の場合、各点の容量をRBG全体に適用
        if self.config.use_rbg_granularity:
            capacity_per_re *= tf.cast(self.rbg_size_sc, self.rdtype)

        # 全レイヤー・全サブキャリアの容量を合算
        # capacity_per_re: [B, N_UT, N_target, rank]
        throughput_per_user = tf.reduce_sum(capacity_per_re, axis=[-1, -2])
        # MCSとSINRの平均を記録
        mcs_idx_avg = tf.reduce_mean(tf.cast(mcs_idx, tf.float32), axis=[-1, -2])
        sinr_eff_avg = tf.reduce_mean(sinr, axis=[-1, -2])

        return {
            "sinr": sinr,
            "sinr_db": sinr_db,
            "sinr_eff_avg": sinr_eff_avg,
            "throughput_per_user": throughput_per_user,
            "mcs_idx_avg": mcs_idx_avg,
            "rank": tf.fill([B, N_UT], tf.cast(rank, tf.float32)),
        }

    def _record_drop(
        self, hist, drop_idx, results, pl_db, p_tx_watt, p_cmax_dbm, mpr_db
    ):
        """ドロップごとの結果を履歴に記録する"""

        def match_hist_shape(tensor):
            rank = tensor.shape.rank
            if rank is not None and rank > 2:
                tensor = tf.reduce_mean(tensor, axis=list(range(2, rank)))
            return tf.reshape(
                tensor, [self.batch_size, self.num_bs, self.num_ut_per_sector]
            )

        return record_results(
            hist,
            drop_idx,
            sim_failed=False,
            pathloss_serving_cell=match_hist_shape(pl_db),
            num_allocated_re=match_hist_shape(
                tf.fill(
                    [self.batch_size, self.num_ut],
                    float(self.resource_grid.num_effective_subcarriers),
                )
            ),
            tx_power_per_ut=match_hist_shape(p_tx_watt),
            num_decoded_bits=match_hist_shape(results["throughput_per_user"]),
            mcs_index=match_hist_shape(results["mcs_idx_avg"]),
            harq_feedback=match_hist_shape(tf.zeros([self.batch_size, self.num_ut])),
            olla_offset=match_hist_shape(tf.zeros([self.batch_size, self.num_ut])),
            sinr_eff=match_hist_shape(results["sinr_eff_avg"]),
            p_cmax_dbm=match_hist_shape(p_cmax_dbm),
            rank=match_hist_shape(results["rank"]),
            mpr_db=match_hist_shape(mpr_db),
            pf_metric=tf.reshape(
                match_hist_shape(tf.zeros([self.batch_size, self.num_ut])),
                [self.batch_size, self.num_bs, 1, 1, self.num_ut_per_sector],
            ),
        )

    # @tf.function(jit_compile=False)
    def call(self, num_drops, tx_power_dbm):
        """シミュレーションのメインループ（オーケストレーター）"""
        hist = init_result_history(
            self.batch_size, num_drops, self.num_bs, self.num_ut_per_sector
        )

        for drop_idx in range(num_drops):
            # 1. トポロジーのセットアップ
            self._setup_drop_topology(drop_idx)

            # 2. アナログビーム選択
            self._select_analog_beams()

            # 3. デジタル重み計算 (SVD)
            # 粒度は self.precoding_granularity を使用
            w_ut_dig, w_bs_dig, s_srv = self._compute_digital_weights(
                granularity=self.precoding_granularity
            )

            # 4. 送信電力制御
            p_tx_watt, pl_db, mpr_db, p_cmax_dbm = self._apply_power_control(
                tx_power_dbm
            )

            # 5. SINR計算 & Link Adaptation
            results = self._process_sinr_and_la(w_ut_dig, w_bs_dig, s_srv, p_tx_watt)

            # 6. 結果の記録
            hist = self._record_drop(
                hist, drop_idx, results, pl_db, p_tx_watt, p_cmax_dbm, mpr_db
            )

        # 履歴をTensorに変換
        final_hist = {}
        for key in hist:
            if isinstance(hist[key], tf.TensorArray):
                final_hist[key] = hist[key].stack()
            else:
                final_hist[key] = hist[key]

        return final_hist
