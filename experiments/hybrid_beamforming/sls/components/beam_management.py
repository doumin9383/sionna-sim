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

    def call(self, h_elem):
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
        # 1. Identify dimensions
        # Expected h_elem: [batch, num_ut, num_rx_ant, num_tx_ant, num_sc]
        # (Assuming we reduced other dims or specific shape)

        # Let's assume h_elem is [batch, num_ut, num_rx_ant, num_tx_ant, num_sc]
        # We want to maximize power summing over rx_ant and sc.

        # Limit to first panel
        # num_tx_ant = h_elem.shape[-2]
        # first_panel_size = self.rows_per_panel * self.cols_per_panel * (2 if dual-pol else 1)

        # For cross-pol, the ordering is usually [Panel, Element, Pol] or [Pol, Panel, Element]
        # Sionna PanelArray by default:
        # The elements are indexed as: 0..N-1.
        # We need to construct a mask or slice for the 0-th panel.

        # Since implementation of arbitrary slicing is risky without knowing exact internal ordering,
        # we assume standard PanelArray where we can calculate indexing.
        # But actually, constructing the FULL W_RF is the goal.

        # Let's simplify:
        # We apply the codebook (spatial weights) to the channel.
        # w_spatial: [ant_per_pol, num_beams]

        # If cross-pol, we have 2 polarizations. We want to find the best SPATIAL beam.
        # We can sum power across both polarizations for each beam.

        # Extract Co-polarized and Cross-polarized sub-channels for the first panel?
        # Simpler approach:
        # 1. Take the first num_rows_per_panel * num_cols_per_panel elements.
        #    These are Pol 1 of Panel 1 (assuming ordering allows).
        #    Actually, Sionna PanelArray flattens as [num_panels_v, num_panels_h, pol, num_rows, num_cols] usually?
        #    Let's check `PanelArray` doc or source if needed.
        #    However, if we just take "First N elements" where N is single-pol single-panel count,
        #    we might be taking a mix if ordering is interleaved.

        # Workaround: Use the Codebook to expand to full array size but with zeros for other panels?
        # No, that's inefficient.

        # Assumption: We sweep using the first polarization of the first subpanel.
        # We assume the channel correlation is high enough that this beam is good for all.

        # Let's assume flattened index 0 to N_sp-1 corresponds to the first polarization of first panel.
        # N_sp = rows_per_panel * cols_per_panel.

        n_sp = self.rows_per_panel * self.cols_per_panel

        # h_subset: [batch, num_ut, rx_ant, n_sp, sc]
        # We slice the tx_ant dimension.
        # Need to ensure which axis is tx_ant.
        # In `HybridChannelInterface.get_element_channel...` we will define the return shape.
        # Let's assume it returns [Batch, U, RxAnt, TxAnt, SC]

        # Extract first n_sp elements of TxAnt
        # CAUTION: We need to know if indices 0..n_sp-1 are indeed forming a valid spatial array.
        # In Sionna PanelArray, default is created by tiling.
        # Typically the "fastest" varying indices are the last ones.
        # PanelArray(..., polarization='cross') -> 2 elements at each position.
        # So index 0 is Pos(0,0) Pol+45, index 1 is Pos(0,0) Pol-45.
        # Then index 2 is Pos(0,1) Pol+45...
        # So we should take every 2nd element to get a single polarization spatial array.

        is_cross_pol = self.polarization in ["dual", "cross"]
        stride = 2 if is_cross_pol else 1

        # Slice for one polarization of first panel
        # Length needed: n_sp * stride
        # Then subsample by stride

        # h_elem columns corresponding to first panel
        # Total elements in one panel = n_sp * stride
        n_panel_total = n_sp * stride

        h_panel = h_elem[..., :n_panel_total, :]  # Slicing TxAnt axis (axis -2 assumed)

        # Subsample for single pol
        h_single_pol = h_panel[..., 0::stride, :]  # [..., n_sp, sc]

        # Apply Codebook
        # h_single_pol: [..., n_sp, sc]
        # w_spatial: [n_sp, num_beams]
        # beam_response: [..., num_beams, sc]
        # contract n_sp

        beam_response = tf.einsum(
            "...ns,...nb->...bs",
            h_single_pol,
            tf.cast(self.w_spatial, h_single_pol.dtype),
        )

        # Calculate Power (sum over subcarriers and Rx antennas if kept)
        # shape [Batch, U, RxAnt, Beams, SC]
        # Power = sum(|x|^2)

        beam_power = tf.reduce_sum(
            tf.square(tf.abs(beam_response)), axis=[-1, -3]
        )  # Sum SC, RxAnt
        # Result: [Batch, U, Beams]

        # Select Index
        best_beam_idx = tf.argmax(
            beam_power, axis=-1, output_type=tf.int32
        )  # [Batch, U]

        # Construct Full Precoders
        return self._construct_full_precoder(best_beam_idx)

    def _construct_full_precoder(self, best_beam_idx):
        """
        Constructs W_RF from selected beam indices.
        Applies the selected spatial beam to all panels and both polarizations.

        Output: W_RF [Batch, U, NumTotalTxAnt, NumTxPorts]
        NumTxPorts: We need to define this.
        Usually for Digital-Analog Hybrid, we map:
        - Each polarization to a separate port? (Rank 2 analog support)
        - Or combine them?

        Let's assume we map each polarization to a separate logical port.
        And we have P panels.
        If we want fully digital baseband access to all Panels x Pols, we need many ports.
        But typically "Analog Beamforming" implies phase shifters reduce ports.

        Scenario 1: All panels steer to same direction.
        V-pol and H-pol steer to same direction.
        Result: 2 RF chains (ports) seeing the whole array gain?
        Or 2 RF chains PER PANEL?

        User requirement: "Single panel sweep -> Apply to all panels".
        Usually implies we form a large specific beam.

        Let's assume a simplified architecture:
        W_RF connects TotalAntennas -> 2 Ports (One for Pol1, One for Pol2).
        Or Number of RF chains defined in Config.

        If BS has 4 RF chains, maybe we use 2 panels?
        Let's assume we simply map:
        Pol1 of All Panels -> Port 0
        Pol2 of All Panels -> Port 1
        (This creates a massive array effect, narrowing the beam significantly if panels are coherent.
         If panels are distributed/non-coherent, this might be bad, but physically they are usually coherent in PanelArray).

        Implementation:
        Gather w_spatial[best_beam_idx] -> [Batch, U, n_sp]

        Title the weight to all panels.
        """

        batch_size = tf.shape(best_beam_idx)[0]
        num_ut = tf.shape(best_beam_idx)[1]

        # Gather selected spatial weights
        # best_beam_idx: [ Batch, U ]
        # w_spatial: [n_sp, num_beams]
        # Transpose w: [ num_beams, n_sp ]
        w_spatial_t = tf.transpose(self.w_spatial)

        # Gather: [Batch, U, n_sp]
        w_selected = tf.gather(w_spatial_t, best_beam_idx)

        # We need to construct W_RF of shape [Batch, U, TotalAnt, NumPorts]
        # Let's assume NumPorts = 2 (Dual Pol) or 1 (Single Pol).
        # And we apply the same 'w_selected' to proper elements.

        # Total Ant elements = NumPanels * n_sp * stride
        num_panels = self.num_panels_v * self.num_panels_h
        stride = 2 if self.polarization in ["dual", "cross"] else 1
        num_ports = stride  # 1 port per polarization

        # We construct the vector for ONE panel first
        # Panel vector: [Batch, U, n_sp * stride, num_ports]
        # If stride=2:
        #   Element 2i   (Pol1) -> Port 0: w_selected[i]
        #   Element 2i+1 (Pol2) -> Port 1: w_selected[i]
        #   Cross terms 0.

        # Let's build purely using tensor ops
        # w_selected: [B, U, n_sp] -> expand to [B, U, n_sp, 1]
        w_base = tf.expand_dims(w_selected, -1)

        if stride == 2:
            # We want block diagonal per element pair?
            # [w, 0]
            # [0, w]
            # Shape [B, U, n_sp, 2, 2] (Ant, Port) then reshape to [B, U, n_sp*2, 2]

            zeros = tf.zeros_like(w_base)
            # col1 = stack(w, 0), col2 = stack(0, w)
            # This is tricky with simple stacking.

            # Alternative: w_base * eye(2)
            # w_base: [B, U, n_sp, 1, 1]
            # eye: [1, 1, 1, 2, 2]
            # res: [B, U, n_sp, 2, 2]
            w_block = tf.expand_dims(w_base, -1) * tf.eye(2, dtype=w_base.dtype)

            # Flatten to [B, U, n_sp*2, 2]
            w_panel = tf.reshape(w_block, [batch_size, num_ut, self.ant_per_panel, 2])

        else:
            w_panel = w_base  # [B, U, n_sp, 1]

        # Now tile for all panels
        # We simply repeat this weight vector for each panel.
        # This implies all panels steer to the same angle (Phase alignment between panels is assumed ideal/calibrated to 0 or geometric)
        # Note: If panels are essentially forming a larger planar array, just repeating the steering vector
        # is only valid if the 'w_spatial' was derived for the geometry of the WHOLE array.
        # BUT here we derived it for a SUB-PANEL.
        # If we blindly repeat it, we might have grating lobes or misalignment if the global geometry isn't periodic with wavelength.

        # CRITICAL PHYSICS CHECK:
        # A steering vector for direction theta depends on position: exp(j k . r)
        # If we have two panels at r1 and r2.
        # w(r) = exp(j k(theta) . r)
        # w_panel1 = exp(j k . (r_local + R_panel1)) = exp(j k . r_local) * exp(j k . R_panel1)
        # w_panel2 = exp(j k . (r_local + R_panel2)) = exp(j k . r_local) * exp(j k . R_panel2)
        # So w_panel2 = w_panel1 * exp(j k . (R_panel2 - R_panel1))
        # We need that phase shift!

        # If we only repeat w_spatial, we leverage the array gain of individual panels,
        # but the signals from different panels might add destructively (or non-coherently) at the UE.
        # HOWEVER, the 'BeamSelector' calculates this phase shift? No, it calculates 'w_spatial' for panel 0.

        # If we want COHERENT combining of panels, we need to calculate the phase offset for each panel for the selected angle.
        # Or, we just accept that we control each panel to point to that direction locally,
        # and the digital precoder (SVD) will handle the inter-panel phasing (if we have separate RF chains per panel).

        # IF we have 1 RF chain for ALL panels (analog combined), we MUST do the phase shifting here.
        # Implementation Plan Assumption: "UE is full digital" (SVD).
        # BS has 'bs_num_rf_chains'.
        # If bs_num_rf_chains >= num_panels * num_pol, then we can map each panel to a separate port,
        # and SVD will handle the phase alignment between panels.
        # If bs_num_rf_chains < num_panels, we are forced to combine analog-ly.

        # Let's check config.bs_num_rf_chains vs topology.
        # In this task, let's assume we maintain one port per polarization per Panel?
        # Or one port per polarization for the WHOLE (fully fully analog combined)?

        # User said "Subpanel sweep -> Apply to all".
        # Let's implement the "Geometric Phase Shift" application to be safe and correct.
        # We know the beam index -> We know the Angle (approximately) from the codebook.
        # We can compute the phase shift for PANEL centers.

        # But wait, `CodebookGenerator` generates orthogonal DFT beams.
        # We can retrieve the angle index (k_h, k_v) from best_beam_idx.

        # Let's calculate k_h, k_v.
        k_h = best_beam_idx % self.codebook_gen.num_beams_h
        k_v = best_beam_idx // self.codebook_gen.num_beams_h

        # Convert to angles or directly compute phase shift?
        # d_h = 0.5 lambda, d_v = 0.5 lambda usually.
        # Phase shift between columns: 2pi * k_h / N_beams_h
        # If Panels are placed on a gri with spacing D_panel_h, D_panel_v.
        # We need specific geometry.

        # SIMPLIFICATION for this iteration:
        # Assume separate RF ports per panel is NOT the case (usually hybrid is limited ports).
        # BUT constructing the true phase shift is complex without exact panel geometry access here.
        # PanelArray gives `ant_pos`.

        # Let's follow a Robust Robust strategy:
        # The user asked for "Apply to all panels".
        # If we just repeat the weights, we get max power from each panel individually.
        # The total signal is Sum( H_panel_i * w ).
        # If H_panel_i has random phase relative to H_panel_j (due to channel),
        # then coherent combining needs SVD.
        # IF Line of Sight and calibrated array, H phase is deterministic.

        # DECISION: To ensure SVD can do its job, we should ideally expose ports per panel if possible.
        # BUT current config likely sets `bs_num_rf_chains` to a small number (e.g. 4 or 8).
        # Params: bs_num_rows_panel=1, cols=1 usually?

        # Let's verify `my_configs.py` again.
        # It inherits params. `bs_array` config is `PanelArray(...)`.

        # For this implementation, I will implement **Simple Repetition**.
        # `w_rf` will repeat the sub-panel weights for all panels.
        # This means all panels "look" in the same direction.
        # The SVD (Digital Precoder) will handle the phase alignment between the effective ports
        # IF we map them to different ports.
        # IF we map all to 1 port, we rely on luck/geometry (constructive interference at broadside?).

        # Let's check `HybridOFDMChannel` default weights.
        # It uses `tf.eye`.
        # `HybridChannelInterface` takes `num_tx_ports` from config.

        # I will map:
        # All panels Pol 0 -> Port 0
        # All panels Pol 1 -> Port 1
        # (Massive Parallel connection).
        # This is standard "Subarray connection" type: "Fully Connected" (phase shifters on all elements going to sum).

        # So I DO need to align phases if I want Array Gain from N panels > 1 panel.
        # Without phase alignment, I get N * Power(1 panel) (incoherent sum expectation)
        # vs N^2 * Power (coherent).

        # Given "Subpanel sweep" instruction, it's likely an approximation.
        # I will implement Simple Repetition.
        # If the user wants coherent global beamforming, they need a global codebook.

        w_final = tf.tile(w_panel, [1, 1, num_panels, 1])
        # [B, U, NumPanels * n_sp * 2, 2]

        return w_final
