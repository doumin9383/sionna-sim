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
from sionna.phy.channel.tr38901 import UMi, UMa, RMa

# Local Components
from .components.hybrid_channel_interface import HybridChannelInterface
from .components.simplified_link_adaptation import WaterFillingLinkAdaptation
from .components.mpr_model import MPRModel
from .components.power_control import PowerControl
from .components.link_adaptation import MCSLinkAdaptation
from .components.get_stream_management import get_stream_management
from .components.get_hist import init_result_history, record_results
from .components.precoder_utils import expand_precoder

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
    ):
        super().__init__(precision=precision)

        self.config = config
        self.scenario = config.scenario
        self.batch_size = int(config.batch_size)
        self.resource_grid = config.resource_grid
        self.num_ut_per_sector = int(config.num_ut_per_sector)
        self.direction = config.direction
        self.bs_max_power_dbm = config.bs_max_power_dbm  # [dBm]
        self.ut_max_power_dbm = config.ut_max_power_dbm  # [dBm]
        self.coherence_time = tf.cast(config.coherence_time, tf.int32)  # [slots]
        num_cells = get_num_hex_in_grid(config.num_rings)
        self.num_bs = num_cells * 3
        self.num_ut = self.num_bs * self.num_ut_per_sector
        self.num_ut_ant = config.ut_array.num_ant
        self.num_bs_ant = config.bs_array.num_ant
        if config.bs_array.polarization == "dual":
            self.num_bs_ant *= 2
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

        # Assume 1 stream for UT antenna
        self.num_streams_per_ut = config.resource_grid.num_streams_per_tx

        # Set TX-RX pairs via StreamManagement
        self.stream_management = get_stream_management(
            config.direction,
            self.num_rx,
            self.num_tx,
            self.num_streams_per_ut,
            config.num_ut_per_sector,
        )
        # Noise power per subcarrier
        self.no = tf.cast(
            BOLTZMANN_CONSTANT * temperature * config.resource_grid.subcarrier_spacing,
            self.rdtype,
        )

        # Slot duration [sec]
        self.slot_duration = (
            config.resource_grid.ofdm_symbol_duration
            * config.resource_grid.num_ofdm_symbols
        )

        # Initialize channel model based on scenario
        self._setup_channel_model(
            config.scenario,
            config.carrier_frequency,
            o2i_model,
            config.ut_array,
            config.bs_array,
            average_street_width,
            average_building_height,
        )

        # Generate multicell topology
        self._setup_topology(config.num_rings, min_bs_ut_dist, max_bs_ut_dist)

        # Instantiate the Hybrid Channel Interface
        self.channel_interface = HybridChannelInterface(
            channel_model=self.channel_model,
            resource_grid=config.resource_grid,
            tx_array=config.bs_array,  # Mapping bs_array to tx_array
            rx_array=config.ut_array,  # Mapping ut_array to rx_array
            num_tx_ports=config.bs_array.num_ant,
            num_rx_ports=config.ut_array.num_ant,
            precision=self.precision,
        )

        # Instantiate simplified link adaptation (Physics Abstraction for SINR)
        self.phy_abstraction = WaterFillingLinkAdaptation(
            resource_grid=config.resource_grid,
            transmitter=None,
            num_streams_per_tx=self.num_streams_per_ut,
            precision=self.precision,
        )

        # Instantiate SLS components
        self.mpr_model = MPRModel(csv_path=config.mpr_table_path)
        self.power_control = PowerControl(p_power_class=config.ut_max_power_dbm)
        self.mcs_adapter = MCSLinkAdaptation()

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

        # Set topology in channel model
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

    @tf.function(jit_compile=False)
    def call(self, num_slots, tx_power_dbm):
        # Initialize result history
        throughput_history = tf.TensorArray(dtype=self.rdtype, size=num_slots)

        # BS-UT Association
        # [num_ut] -> values are serving BS indices
        serving_bs_idx = tf.argmax(self.stream_management.rx_tx_association, axis=1)

        # --------------- #
        # Simulate a slot #
        # --------------- #
        def simulate_slot(slot, throughput_history):
            # 1. Get Full Channel Information (For Interference/Evaluation)
            # h: [batch, num_ut, num_bs, ofdm, sc, rx_ports, tx_ports]
            h, _, u_all, _ = self.channel_interface.get_full_channel_info(
                self.batch_size
            )

            # 2. Get Precoding Channel (For Beamforming Calculation)
            # h_prec: [batch, num_ut, num_bs, ofdm, num_blocks, rx_ports, tx_ports]
            h_prec = self.channel_interface.get_precoding_channel(
                self.batch_size,
                granularity=self.precoding_granularity,
                rbg_size_sc=self.rbg_size_sc,
            )

            # Compute SVD on Coarse Channel
            s_prec, u_prec, v_prec = tf.linalg.svd(h_prec)

            # Expand v_prec to Full Bandwidth
            # v_prec: [batch, num_ut, num_bs, ofdm, num_blocks, tx_ports, tx_ports]
            # We need to expand dim -3 (num_blocks) to num_sc
            total_subcarriers = self.resource_grid.num_effective_subcarriers
            v_expanded = expand_precoder(
                v_prec,
                total_subcarriers=total_subcarriers,
                granularity_type=self.precoding_granularity,
                rbg_size_sc=self.rbg_size_sc,
            )

            # 3. Extract Serving Precoders and Combiners
            serving_bs_idx_batched = tf.broadcast_to(
                serving_bs_idx, [self.batch_size, self.num_ut]
            )

            # Extract serving U (Ideal) and V (Granular)
            # u_serv: [batch, num_ut, ofdm, sc, rx_ports, rx_ports] (Full resolution)
            # v_serv: [batch, num_ut, ofdm, sc, tx_ports, tx_ports] (Expanded granular)
            u_serv = tf.gather(u_all, serving_bs_idx_batched, axis=2, batch_dims=2)
            v_serv = tf.gather(v_expanded, serving_bs_idx_batched, axis=2, batch_dims=2)

            # 4. Calculate Interference (The "Box" for future extensions)
            # BS_precoders: Assuming UT i is served by BS i (for num_ut_per_sector=1).
            bs_precoders = v_serv

            # a. UT i's combiner applied to all BS links: H_u = U_i^H * H_ij
            # [batch, ut, bs, ofdm, sc, streams, tx_p]
            h_u = tf.einsum("buosrp,bujosrt->bujospt", tf.math.conj(u_serv), h)

            # b. BS j's precoder applied: H_eff = H_u * V_j
            # [batch, ut, bs, ofdm, sc, streams_i, streams_j]
            h_eff = tf.einsum("bujospt,bjostq->bujospq", h_u, bs_precoders)

            # c. Interference summation
            # Interference power to user i: sum over j != serving_bs_idx[i] of |h_eff_ij|^2
            interference_per_bs = tf.reduce_sum(tf.square(tf.abs(h_eff)), axis=-1)

            # Mask out serving link
            mask = tf.one_hot(serving_bs_idx, depth=self.num_bs)
            mask = tf.reshape(mask, [1, self.num_ut, self.num_bs, 1, 1, 1])

            interference_total = tf.reduce_sum(
                interference_per_bs * (1.0 - mask), axis=2
            )

            # Effective Noise per stream: N0 + Interference
            noise_plus_interference = self.no + interference_total

            # Calculate Effective Channel Gains (s_serv) from h_eff
            # This captures beamforming mismatch due to granularity
            # h_eff: [batch, ut, bs, ofdm, sc, stream, stream] -> Gather serving BS
            h_eff_serv = tf.gather(h_eff, serving_bs_idx_batched, axis=2, batch_dims=2)
            # Take diagonal (signal power on streams)
            s_serv = tf.abs(tf.linalg.diag_part(h_eff_serv))

            # 4. Power Control & Link Adaptation
            # a. Calculate Path Loss (Simple Euclidean distance based approximation for PC)
            # ut_loc: [batch, num_ut, 3]
            # serving_bs_idx: [batch, num_ut]
            # bs_loc: [batch, num_bs, 3]
            serving_bs_loc = tf.gather(self.bs_loc, serving_bs_idx, batch_dims=1)
            dist = tf.norm(self.ut_loc - serving_bs_loc, axis=-1)  # [batch, num_ut]

            # Simple UMi Path Loss Model for 3.5GHz (Placeholder)
            # PL = 28.0 + 22*log10(d) + 20*log10(fc)
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
                # Downlink: Use fixed power (split among streams/users handled in Power Allocation?)
                # For now, use the passed tx_power_dbm argument
                # But tx_power_dbm is scalar/tensor? call(..., tx_power_dbm)
                # If scalar, broadcast to users?
                # In DL, total BS power is split.
                # Here simplified: Assume tx_power_dbm is per-link or per-user equivalent?
                # Or total BS power?
                # Using the argument passed to call()
                p_tx_dbm = tx_power_dbm

            # Broadcast p_tx_dbm to [batch, num_ut] if it calculated scalar/vector
            # p_tx_dbm might be tensor [batch, num_ut]
            total_power = dbm_to_watt(p_tx_dbm)

            # Reshape/Broadcast for broadcasting: [batch, num_ut, 1, 1, 1]
            if len(total_power.shape) == 0:  # Scalar
                total_power = tf.broadcast_to(
                    total_power, [self.batch_size, self.num_ut, 1, 1, 1]
                )
            elif len(total_power.shape) == 2:  # [batch, num_ut]
                total_power = tf.reshape(
                    total_power, [self.batch_size, self.num_ut, 1, 1, 1]
                )
            elif len(total_power.shape) == 3:  # [batch, num_ut, 1]
                total_power = tf.reshape(
                    total_power, [self.batch_size, self.num_ut, 1, 1, 1]
                )
            else:
                # Attempt to broadcast/reshape if dimensions allow, safeguard
                total_power = tf.reshape(
                    total_power, [self.batch_size, self.num_ut, 1, 1, 1]
                )

            # d. Physics Abstraction (Water Filling -> SINR)
            p_alloc, sinr = self.phy_abstraction.call(
                s_serv, noise_plus_interference, total_power
            )

            # e. MCS Selection & Throughput
            # MCS Adapter expects SINR in dB
            sinr_db = 10.0 * tf.math.log(tf.maximum(sinr, 1e-20)) / tf.math.log(10.0)

            # Use Discrete MCS Table Lookup
            # Returns Spectral Efficiency (bits/symbol) including BLER penalty
            capacity_per_re = self.mcs_adapter.get_throughput_vectorized(sinr_db)

            throughput_per_user = tf.reduce_sum(capacity_per_re, axis=[-1, -2])

            # Store and Update
            throughput_history = throughput_history.write(slot, throughput_per_user)
            self.ut_loc = self.ut_loc + self.ut_velocities * self.slot_duration
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

            return slot + 1, throughput_history

        # Run loop
        _, final_history = tf.while_loop(
            lambda i, *_: i < num_slots,
            simulate_slot,
            [0, throughput_history],
        )

        return final_history.stack()
