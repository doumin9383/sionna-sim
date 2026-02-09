import tensorflow as tf
from sionna.phy import Block, PI


class CodebookGenerator(Block):
    """
    Generates DFT Codebook for Analog Beamforming.
    Supports single panel codebook generation.
    """

    def __init__(
        self,
        num_rows_per_panel,
        num_cols_per_panel,
        polarization="cross",
        oversampling_factor=1,
        dtype=tf.complex64,
    ):
        super().__init__(dtype=dtype)
        self.num_rows = num_rows_per_panel
        self.num_cols = num_cols_per_panel
        self.polarization = polarization
        self.oversampling_factor = oversampling_factor

        # Calculate number of beams
        self.num_beams_h = self.num_cols * oversampling_factor
        self.num_beams_v = self.num_rows * oversampling_factor
        self.total_beams = self.num_beams_h * self.num_beams_v

    def call(self):
        """
        Generates DFT codebook for a single polarization.
        Returns: [num_ant_per_pol, total_beams]
        """
        # DFT Vectors for Horizontal (Azimuth)
        # n = 0...N-1, k = 0...K-1
        # w_k(n) = exp(j * 2pi * n * k / K)
        n_h = tf.range(self.num_cols, dtype=tf.float32)
        k_h = tf.range(self.num_beams_h, dtype=tf.float32)

        # [N, 1] * [1, K] -> [N, K]
        # Standard DFT definition uses exp(-j...), but beamsteering usually uses exp(j...) to compensate channel phase exp(-j...)
        # channel phase delay: exp(-j * k * d)
        # beamformer: exp(j * k * d)
        # We use standard DFT matrix definition which is orthogonal.
        # W_nk = exp(-j * 2pi * n * k / N_fft)
        # For beamforming 0 to pi, we map indices.

        # Simple DFT Codebook:
        weights_h = tf.complex(
            tf.math.cos(
                2.0
                * PI
                * tf.expand_dims(n_h, -1)
                * tf.expand_dims(k_h, 0)
                / self.num_beams_h
            ),
            tf.math.sin(
                2.0
                * PI
                * tf.expand_dims(n_h, -1)
                * tf.expand_dims(k_h, 0)
                / self.num_beams_h
            ),
        )

        # DFT Vectors for Vertical (Elevation)
        n_v = tf.range(self.num_rows, dtype=tf.float32)
        k_v = tf.range(self.num_beams_v, dtype=tf.float32)

        weights_v = tf.complex(
            tf.math.cos(
                2.0
                * PI
                * tf.expand_dims(n_v, -1)
                * tf.expand_dims(k_v, 0)
                / self.num_beams_v
            ),
            tf.math.sin(
                2.0
                * PI
                * tf.expand_dims(n_v, -1)
                * tf.expand_dims(k_v, 0)
                / self.num_beams_v
            ),
        )

        # Kronecker Product to get 2D Array Response
        # [Nv, Kv] x [Nh, Kh] -> [Nv, Nh, Kv, Kh] -> [Nv*Nh, Kv*Kh]
        # We want flattend antenna dimension and flattened beam dimension

        # Expand dims for broadcasting
        # w_v: [Nv, 1, Kv, 1]
        w_v_exp = tf.expand_dims(tf.expand_dims(weights_v, axis=1), axis=3)
        # w_h: [1, Nh, 1, Kh]
        w_h_exp = tf.expand_dims(tf.expand_dims(weights_h, axis=0), axis=2)

        # Product: [Nv, Nh, Kv, Kh]
        w_2d = w_v_exp * w_h_exp

        # Reshape to [num_ant_per_pol, total_beams]
        w_flat = tf.reshape(w_2d, [self.num_rows * self.num_cols, self.total_beams])

        # Normalize
        w_flat = w_flat / tf.sqrt(
            tf.cast(self.num_rows * self.num_cols, dtype=w_flat.dtype)
        )

        return w_flat

    def get_dual_pol_codebook(self):
        """
        Returns codebook for dual polarization.
        Assumes co-phasing is handled by digital precoder or fixed.
        Here we simply apply the same beam to both polarizations (block diagonal).

        Returns: [num_ant_total, total_beams] (Note: rank 1 per beam direction)
        Or should we return [num_ant_total, total_beams * 2] allowing independent selection?

        For analog beamforming in 5G, usually the same spatial beam is applied to both polarizations.
        And we have 2 ports per beam (V-pol port, H-pol port).

        So W_RF mapping:
        Input Ports: 2 * num_beams (if we expose all beams as ports)
        Or Selected Ports: 2 (for 1 selected beam direction)

        This CodebookGenerator generates the spatial weights for one polarization.
        """
        return self.call()


class BeamSelector(Block):
    """
    Selects the best analog beam for BS.
    Strategy: "Sub-panel Sweep"
    1. Extract channel corresponding to the first sub-panel.
    2. Apply DFT codebook to this sub-channel.
    3. Select best beam index based on received power / singular value.
    4. Construct full W_RF by applying the selected beam weight to all sub-panels.
    """

    def __init__(
        self,
        num_rows_per_panel,
        num_cols_per_panel,
        num_panels_v,
        num_panels_h,
        polarization,
        oversampling_factor=1,
        dtype=tf.complex64,
    ):
        super().__init__(dtype=dtype)

        self.rows_per_panel = num_rows_per_panel
        self.cols_per_panel = num_cols_per_panel
        self.num_panels_v = num_panels_v
        self.num_panels_h = num_panels_h
        self.polarization = polarization

        self.ant_per_panel = self.rows_per_panel * self.cols_per_panel
        if self.polarization in ["dual", "cross"]:
            self.ant_per_panel *= 2

        # Create Codebook Generator
        self.codebook_gen = CodebookGenerator(
            self.rows_per_panel,
            self.cols_per_panel,
            polarization=self.polarization,
            oversampling_factor=oversampling_factor,
            dtype=dtype,
        )

        # Pre-compute codebook
        # We generate spatial weights for a single polarization layer
        # w_spatial: [ant_per_pol_in_panel, num_beams]
        self.w_spatial = self.codebook_gen.call()
        self.num_beams = self.w_spatial.shape[1]

    def _extract_subpanel_channel(self, h_elem):
        """
        Extracts channel corresponding to the first top-left panel (index 0,0).
        h_elem: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, ... ] (Up to rank 7)
        Assuming 'h_elem' is the full element-wise channel.
        We need to identify which 'tx_ant' indices correspond to the first panel.

        PanelArray element ordering (Sionna 0.16+):
        Usually: [Pol, Panel_V, Panel_H, El_V, El_H] flattened.
        Let's check PanelArray implementation details or assume standard ordering.

        If we can't be sure, we need to look at bs_array.ant_pos or similar.
        However, for a regular PanelArray, the first N elements usually correspond to the first panel
        if created with default ordering.

        Number of elements in first panel = rows_per_panel * cols_per_panel * pol
        """
        # Slicing the TX antenna dimension.
        # h_elem shape is typically [batch, num_ut, 1, 1, num_bs_ant, 1, sc] (from previous logs)
        # Wait, get_element_channel might return [batch, ut, rx_ant, tx_ant, sc] or something.
        # Let's verify shape in 'call'.

        return h_elem

    def call(self, h_elem, antenna_array):
        """
        Selects best beam for each user/link based on h_elem.

        Args:
            h_elem (tf.Tensor): Element-domain channel.
                Shape: [batch, num_ut, num_rx_cp, 1, num_tx_ant, 1, num_sc] (approx)
                We assume 'num_tx_ant' is the dimension to sweep.
                Note: h_elem from `get_element_channel_for_beam_selection` needs to be defined.

        Returns:
            w_rf (tf.Tensor): Analog precoder weights [batch, num_ut, num_tx_ant, num_tx_ports]
                Constructed for all panels.
        """
        # --- 1. パネル0のチャネル抽出 ---
        n = 0  # first panel
        # 1パネルあたりのアンテナ数 (偏波含む)
        n_ant_panel = antenna_array._num_panel_ant.numpy()
        # パネル番号 'n' のアンテナインデックスの範囲
        start_idx = n * n_ant_panel
        end_idx = (n + 1) * n_ant_panel
        # パネル 'n' の物理座標を確認
        # panel_n_pos = antenna_array.ant_pos[start_idx:end_idx]
        # Get h_elem for first panel
        h_panel = h_elem[..., start_idx:end_idx, :]

        # --- 2. コードブックをDual-pol構造に拡張 ---
        # w_spatial: [ant_per_pol(rows*cols), num_beams]
        # Sionnaのインデックスに合わせて [Pol1, Pol2, Pol1, Pol2...] にリピート
        # w_dual_pol: [ant_per_panel(rows*cols*2), num_beams]
        w_dual_pol = tf.repeat(self.w_spatial, repeats=2, axis=0)

        # --- 3. ビームスイープ (全ビームの応答計算) ---
        # h_panel: [B, U, R, N_ant_panel, S]
        # w_dual_pol: [N_ant_panel, N_beams]
        # beam_response: [B, U, R, N_beams, S]
        beam_response = tf.einsum(
            "...ns,nk->...ks", h_panel, tf.cast(w_dual_pol, h_panel.dtype)
        )

        # --- 4. パワー計算とベストビーム選択 ---
        # Rxアンテナ(axis=2)とサブキャリア(axis=-1)で平均/和をとる
        beam_power = tf.reduce_sum(tf.abs(beam_response) ** 2, axis=[2, -1])
        best_beam_idx = tf.argmax(beam_power, axis=-1, output_type=tf.int32)  # [B, U]

        # --- 5. 全パネルへのブロードキャスト適用 ---
        # _construct_full_precoder 内で全パネル(32個)に対して
        # 選択されたビームをブロック対角に配置する
        return self._construct_full_precoder(best_beam_idx)

    def _construct_full_precoder(self, best_beam_idx):
        """
        best_beam_idx: [Batch, User]
        Returns: [Batch, User, Total_Ant(num_panels*4), Total_RF(num_panels*2)]
        """
        # 1. 選択された空間重みを取得 [Batch, U, ant_per_pol_in_panel(2)]
        # w_spatial.T は [num_beams, 2]
        w_selected = tf.gather(tf.transpose(self.w_spatial), best_beam_idx)

        batch_size = tf.shape(best_beam_idx)[0]
        num_ut = tf.shape(best_beam_idx)[1]
        num_panels = self.num_panels_v * self.num_panels_h  # 32

        # 2. パネル単位のウェイト行列作成 [Batch, U, 4(Ant), 2(Port)]
        # Sionnaの順序 [P1, P2, P1, P2] に合わせる
        # w_selected: [B, U, 2] -> w0, w1
        w0 = w_selected[:, :, 0:1]  # [B, U, 1]
        w1 = w_selected[:, :, 1:2]  # [B, U, 1]
        z = tf.zeros_like(w0)

        # Port0用 (Pol1のみに重み): [w0, 0, w1, 0]
        port0 = tf.stack([w0, z, w1, z], axis=-2)  # [B, U, 4, 1]
        # Port1用 (Pol2のみに重み): [0, w0, 0, w1]
        port1 = tf.stack([z, w0, z, w1], axis=-2)  # [B, U, 4, 1]

        # パネル内接続行列 [B, U, 4, 2]
        w_panel = tf.concat([port0, port1], axis=-1)

        # 3. 全パネルをブロック対角に配置
        # 各パネルが独立したRFポート(2個ずつ)を持つように拡張
        # identity: [num_panels, num_panels]
        eye_panels = tf.eye(num_panels, dtype=self.dtype)

        # [B, U, 1, 1, 4, 2] * [1, 1, P, P, 1, 1] -> [B, U, P, P, 4, 2]
        w_big = tf.expand_dims(tf.expand_dims(w_panel, 2), 2) * tf.reshape(
            eye_panels, [1, 1, num_panels, num_panels, 1, 1]
        )

        # 次元置換と結合: [B, U, P(row), Ant, P(col), Port] -> [B, U, P*Ant, P*Port]
        w_rf = tf.transpose(w_big, [0, 1, 2, 4, 3, 5])
        w_rf = tf.reshape(w_rf, [batch_size, num_ut, num_panels * 4, num_panels * 2])

        return w_rf
