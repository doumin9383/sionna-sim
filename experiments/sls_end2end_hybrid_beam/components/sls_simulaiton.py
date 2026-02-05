import sionna
import tensorflow as tf
# Additional external libraries
import matplotlib.pyplot as plt
import numpy as np

# Sionna components
from sionna.sys.utils import spread_across_subcarriers
from sionna.sys import (
    PHYAbstraction,
    OuterLoopLinkAdaptation,
    gen_hexgrid_topology,
    get_pathloss,
    open_loop_uplink_power_control,
    downlink_fair_power_control,
    get_num_hex_in_grid,
    PFSchedulerSUMIMO,
)
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.utils import db_to_lin, dbm_to_watt, log2, insert_dims
from sionna.phy import config, dtypes, Block
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, PanelArray
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid,
    RZFPrecodedChannel,
    EyePrecodedChannel,
    LMMSEPostEqualizationSINR,
)

# Local Components
from .channel_matrix import ChannelMatrix
from .get_stream_management import get_stream_management
from .get_sinr import get_sinr
from .estimate_achivable_rate import estimate_achievable_rate
from .get_hist import init_result_history, record_results

# Set random seed for reproducibility
sionna.phy.config.seed = 42

# Internal computational precision
sionna.phy.config.precision = "single"  # 'single' or 'double'


class SystemLevelSimulator(Block):
    def __init__(
        self,
        batch_size,
        num_rings,
        num_ut_per_sector,
        carrier_frequency,
        resource_grid,
        scenario,
        direction,
        ut_array,
        bs_array,
        bs_max_power_dbm,
        ut_max_power_dbm,
        coherence_time,
        pf_beta=0.98,
        max_bs_ut_dist=None,
        min_bs_ut_dist=None,
        temperature=294,
        o2i_model="low",
        average_street_width=20.0,
        average_building_height=5.0,
        precision=None,
    ):
        super().__init__(precision=precision)

        assert scenario in ["umi", "uma", "rma"]
        assert direction in ["uplink", "downlink"]
        self.scenario = scenario
        self.batch_size = int(batch_size)
        self.resource_grid = resource_grid
        self.num_ut_per_sector = int(num_ut_per_sector)
        self.direction = direction
        self.bs_max_power_dbm = bs_max_power_dbm  # [dBm]
        self.ut_max_power_dbm = ut_max_power_dbm  # [dBm]
        self.coherence_time = tf.cast(coherence_time, tf.int32)  # [slots]
        num_cells = get_num_hex_in_grid(num_rings)
        self.num_bs = num_cells * 3
        self.num_ut = self.num_bs * self.num_ut_per_sector
        self.num_ut_ant = ut_array.num_ant
        self.num_bs_ant = bs_array.num_ant
        if bs_array.polarization == "dual":
            self.num_bs_ant *= 2
        if self.direction == "uplink":
            self.num_tx, self.num_rx = self.num_ut, self.num_bs
            self.num_tx_ant, self.num_rx_ant = self.num_ut_ant, self.num_bs_ant
            self.num_tx_per_sector = self.num_ut_per_sector
        else:
            self.num_tx, self.num_rx = self.num_bs, self.num_ut
            self.num_tx_ant, self.num_rx_ant = self.num_bs_ant, self.num_ut_ant
            self.num_tx_per_sector = 1

        # Assume 1 stream for UT antenna
        self.num_streams_per_ut = resource_grid.num_streams_per_tx

        # Set TX-RX pairs via StreamManagement
        self.stream_management = get_stream_management(
            direction,
            self.num_rx,
            self.num_tx,
            self.num_streams_per_ut,
            num_ut_per_sector,
        )
        # Noise power per subcarrier
        self.no = tf.cast(
            BOLTZMANN_CONSTANT * temperature * resource_grid.subcarrier_spacing,
            self.rdtype,
        )

        # Slot duration [sec]
        self.slot_duration = (
            resource_grid.ofdm_symbol_duration * resource_grid.num_ofdm_symbols
        )

        # Initialize channel model based on scenario
        self._setup_channel_model(
            scenario,
            carrier_frequency,
            o2i_model,
            ut_array,
            bs_array,
            average_street_width,
            average_building_height,
        )

        # Generate multicell topology
        self._setup_topology(num_rings, min_bs_ut_dist, max_bs_ut_dist)

        # Instantiate a PHY abstraction object
        self.phy_abs = PHYAbstraction(precision=self.precision)

        # Instantiate a link adaptation object
        self.olla = OuterLoopLinkAdaptation(
            self.phy_abs,
            self.num_ut_per_sector,
            batch_size=[self.batch_size, self.num_bs],
        )

        # Instantiate a scheduler object
        self.scheduler = PFSchedulerSUMIMO(
            self.num_ut_per_sector,
            resource_grid.fft_size,
            resource_grid.num_ofdm_symbols,
            batch_size=[self.batch_size, self.num_bs],
            num_streams_per_ut=self.num_streams_per_ut,
            beta=pf_beta,
            precision=self.precision,
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
        """G enerate and set up network topology"""
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

    def _reset(self, bler_target, olla_delta_up):
        """Reset OLLA and HARQ/SINR feedback"""
        # Link Adaptation
        self.olla.reset()
        self.olla.bler_target = bler_target
        self.olla.olla_delta_up = olla_delta_up

        # HARQ feedback (no feedback, -1)
        last_harq_feedback = -tf.ones(
            [self.batch_size, self.num_bs, self.num_ut_per_sector], dtype=tf.int32
        )

        # SINR feedback
        sinr_eff_feedback = tf.ones(
            [self.batch_size, self.num_bs, self.num_ut_per_sector], dtype=self.rdtype
        )

        # N. decoded bits
        num_decoded_bits = tf.zeros(
            [self.batch_size, self.num_bs, self.num_ut_per_sector], tf.int32
        )
        return last_harq_feedback, sinr_eff_feedback, num_decoded_bits

    def _group_by_sector(self, tensor):
        """Group tensor by sector
        - Input: [batch_size, num_ut, num_ofdm_symbols]
        - Output: [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
        """
        tensor = tf.reshape(
            tensor,
            [
                self.batch_size,
                self.num_bs,
                self.num_ut_per_sector,
                self.resource_grid.num_ofdm_symbols,
            ],
        )
        # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
        return tf.transpose(tensor, [0, 1, 3, 2])

    @tf.function(jit_compile=True)
    def call(
        self,
        num_slots,
        alpha_ul,
        p0_dbm_ul,
        bler_target,
        olla_delta_up,
        mcs_table_index=1,
        fairness_dl=0,
        guaranteed_power_ratio_dl=0.5,
    ):

        # -------------- #
        # Initialization #
        # -------------- #
        # Initialize result history
        hist = init_result_history(
            self.batch_size, num_slots, self.num_bs, self.num_ut_per_sector
        )

        # Reset OLLA and HARQ/SINR feedback
        last_harq_feedback, sinr_eff_feedback, num_decoded_bits = self._reset(
            bler_target, olla_delta_up
        )

        # Initialize channel matrix
        self.channel_matrix = ChannelMatrix(
            self.resource_grid,
            self.batch_size,
            self.num_rx,
            self.num_tx,
            self.coherence_time,
            precision=self.precision,
        )
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym,
        #  num_subcarriers]
        h_freq = self.channel_matrix(self.channel_model)

        # --------------- #
        # Simulate a slot #
        # --------------- #
        def simulate_slot(
            slot, hist, harq_feedback, sinr_eff_feedback, num_decoded_bits, h_freq
        ):
            try:
                # ------- #
                # Channel #
                # ------- #
                # Update channel matrix
                h_freq = self.channel_matrix.update(self.channel_model, h_freq, slot)

                # Apply fading
                h_freq_fading = self.channel_matrix.apply_fading(h_freq)

                # --------- #
                # Scheduler #
                # --------- #
                # Estimate achievable rate
                # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector]
                rate_achievable_est = estimate_achievable_rate(
                    self.olla.sinr_eff_db_last,
                    self.resource_grid.num_ofdm_symbols,
                    self.resource_grid.fft_size,
                )

                # SU-MIMO Proportional Fairness scheduler
                # [batch_size, num_bs, num_ofdm_sym, num_subcarriers,
                #  num_ut_per_sector, num_streams_per_ut]
                is_scheduled = self.scheduler(num_decoded_bits, rate_achievable_est)

                # N. allocated subcarriers
                num_allocated_sc = tf.minimum(
                    tf.reduce_sum(tf.cast(is_scheduled, tf.int32), axis=-1), 1
                )
                # [batch_size, num_bs, num_ofdm_sym, num_ut_per_sector]
                num_allocated_sc = tf.reduce_sum(num_allocated_sc, axis=-2)

                # N. allocated resources per slot
                # [batch_size, num_bs, num_ut_per_sector]
                num_allocated_re = tf.reduce_sum(
                    tf.cast(is_scheduled, tf.int32), axis=[-1, -3, -4]
                )

                # ------------- #
                # Power control #
                # ------------- #
                # Compute pathloss
                # [batch_size, num_rx, num_tx, num_ofdm_symbols], [batch_size, num_ut, num_ofdm_symbols]
                pathloss_all_pairs, pathloss_serving_cell = get_pathloss(
                    h_freq_fading,
                    rx_tx_association=tf.convert_to_tensor(
                        self.stream_management.rx_tx_association
                    ),
                )
                # Group by sector
                # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
                pathloss_serving_cell = self._group_by_sector(pathloss_serving_cell)

                if self.direction == "uplink":
                    # Open-loop uplink power control
                    # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
                    tx_power_per_ut = open_loop_uplink_power_control(
                        pathloss_serving_cell,
                        num_allocated_sc,
                        alpha=alpha_ul,
                        p0_dbm=p0_dbm_ul,
                        ut_max_power_dbm=self.ut_max_power_dbm,
                    )
                else:
                    # Channel quality estimation:
                    # Estimate interference from neighboring base stations
                    # [batch_size, num_ut, num_ofdm_symbols]

                    one = tf.cast(1, pathloss_serving_cell.dtype)

                    # Total received power
                    # [batch_size, num_ut, num_ofdm_symbols]
                    rx_power_tot = tf.reduce_sum(one / pathloss_all_pairs, axis=-2)
                    # [batch_size, num_bs, num_ut_per_sector, num_ofdm_symbols]
                    rx_power_tot = self._group_by_sector(rx_power_tot)

                    # Interference from neighboring base stations
                    interference_dl = rx_power_tot - one / pathloss_serving_cell
                    interference_dl *= dbm_to_watt(self.bs_max_power_dbm)

                    # Fair downlink power allocation
                    # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
                    tx_power_per_ut, _ = downlink_fair_power_control(
                        pathloss_serving_cell,
                        interference_dl + self.no,
                        num_allocated_sc,
                        bs_max_power_dbm=self.bs_max_power_dbm,
                        guaranteed_power_ratio=guaranteed_power_ratio_dl,
                        fairness=fairness_dl,
                        precision=self.precision,
                    )

                # For each user, distribute the power uniformly across
                # subcarriers and streams
                # [batch_size, num_bs, num_tx_per_sector,
                #  num_streams_per_tx, num_ofdm_sym, num_subcarriers]
                tx_power = spread_across_subcarriers(
                    tx_power_per_ut,
                    is_scheduled,
                    num_tx=self.num_tx_per_sector,
                    precision=self.precision,
                )

                # --------------- #
                # Per-stream SINR #
                # --------------- #
                # [batch_size, num_bs, num_ofdm_sym, num_subcarriers,
                #  num_ut_per_sector, num_streams_per_ut]
                sinr = get_sinr(
                    tx_power,
                    self.stream_management,
                    self.no,
                    self.direction,
                    h_freq_fading,
                    self.num_bs,
                    self.num_ut_per_sector,
                    self.num_streams_per_ut,
                    self.resource_grid,
                )

                # --------------- #
                # Link adaptation #
                # --------------- #
                # [batch_size, num_bs, num_ut_per_sector]
                mcs_index = self.olla(
                    num_allocated_re,
                    harq_feedback=harq_feedback,
                    sinr_eff=sinr_eff_feedback,
                )

                # --------------- #
                # PHY abstraction #
                # --------------- #
                # [batch_size, num_bs, num_ut_per_sector]
                num_decoded_bits, harq_feedback, sinr_eff, _, _ = self.phy_abs(
                    mcs_index,
                    sinr=sinr,
                    mcs_table_index=mcs_table_index,
                    mcs_category=int(self.direction == "downlink"),
                )

                # ------------- #
                # SINR feedback #
                # ------------- #
                # [batch_size, num_bs, num_ut_per_sector]
                sinr_eff_feedback = tf.where(
                    num_allocated_re > 0, sinr_eff, tf.cast(0.0, self.rdtype)
                )

                # Record results
                hist = record_results(
                    hist,
                    slot,
                    sim_failed=False,
                    pathloss_serving_cell=tf.reduce_sum(pathloss_serving_cell, axis=-2),
                    num_allocated_re=num_allocated_re,
                    tx_power_per_ut=tf.reduce_sum(tx_power_per_ut, axis=-2),
                    num_decoded_bits=num_decoded_bits,
                    mcs_index=mcs_index,
                    harq_feedback=harq_feedback,
                    olla_offset=self.olla.offset,
                    sinr_eff=sinr_eff,
                    pf_metric=self.scheduler.pf_metric,
                )

            except tf.errors.InvalidArgumentError as e:
                print(
                    f"SINR computation did not succeed at slot {slot}.\n"
                    f"Error message: {e}. Skipping slot..."
                )
                hist = record_results(
                    hist,
                    slot,
                    shape=[self.batch_size, self.num_bs, self.num_ut_per_sector],
                    sim_failed=True,
                )

            # ------------- #
            # User mobility #
            # ------------- #
            self.ut_loc = self.ut_loc + self.ut_velocities * self.slot_duration

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

            return [
                slot + 1,
                hist,
                harq_feedback,
                sinr_eff_feedback,
                num_decoded_bits,
                h_freq,
            ]

        # --------------- #
        # Simulation loop #
        # --------------- #
        _, hist, *_ = tf.while_loop(
            lambda i, *_: i < num_slots,
            simulate_slot,
            [0, hist, last_harq_feedback, sinr_eff_feedback, num_decoded_bits, h_freq],
        )

        for key in hist:
            hist[key] = hist[key].stack()
        return hist
