import tensorflow as tf
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.sls_end2end.components.sls_simulaiton import SystemLevelSimulator
from experiments.sls_end2end.components.channel_matrix import ChannelMatrix
from experiments.sls_end2end.components.get_hist import (
    init_result_history,
    record_results,
)
from experiments.sls_end2end.components.estimate_achivable_rate import (
    estimate_achievable_rate,
)
from experiments.sls_end2end.components.get_sinr import get_sinr
from experiments.sls_end2end.components.get_stream_management import (
    get_stream_management,
)

from sionna.sys import (
    get_pathloss,
    open_loop_uplink_power_control,
    downlink_fair_power_control,
)
from sionna.sys.utils import spread_across_subcarriers
from sionna.phy.utils import dbm_to_watt, insert_dims
from sionna.phy import Block
from sionna.phy import config

from experiments.sls_end2end_hybrid_beam.hybrid_channels import HybridOFDMChannel


class HybridChannelMatrix(ChannelMatrix):
    """
    Adapts ChannelMatrix to use HybridOFDMChannel (Port-based).
    """

    def __init__(
        self,
        hybrid_channel,
        batch_size,
        num_rx,
        num_tx_ant,
        num_rx_ant,
        coherence_time,
        precision=None,
    ):
        # Note: num_tx/rx in super() are actually num_tx, num_rx, NOT ant counts in some contexts?
        # Check ChannelMatrix.__init__: (self, resource_grid, batch_size, num_rx, num_tx, coherence_time)
        # It uses num_rx, num_tx for fading initialization [batch, num_rx, num_tx].
        # In HBF, fading should probably be applied per port or per link?
        # Standard ChannelMatrix applies fading per Rx-Tx link (path).

        # We need to pass the "Hybrid channel" instance or factory
        self.hybrid_channel = hybrid_channel

        # Pass dummy resource_grid as it's in hybrid_channel
        super().__init__(
            hybrid_channel.resource_grid,
            batch_size,
            num_rx,
            num_tx_ant,
            coherence_time,
            precision,
        )
        # Wait, super().__init__ uses num_rx, num_tx to init fading:
        # self.rho_fading = ... [batch, num_rx, num_tx]
        # BUT here we want fading per link?
        # In Sionna SLS, num_tx usually means "Number of Transmitters" (Users), not antennas.
        # Check SLS usage: self.num_tx = self.num_ut (Uplink).
        # So it is correct.

    def call(self, channel_model_unused=None):
        # We override call to use hybrid_channel.get_port_channel
        # Ignore channel_model argument as self.hybrid_channel has it.
        # Return H_port [batch, num_rx, num_rx_port, num_tx, num_tx_port, ...]

        # Assuming num_tx_ports/num_rx_ports are defined in hybrid_channel or passed.
        # For now, we assume hybrid_channel knows it or we pass it?
        # hybrid_channel.get_port_channel(batch_size, num_tx_ports, ...)

        # We need to know port counts.
        # Let's assume 1 port per polarization for MVP? Or configurable.
        # Let's extract from tx_array/rx_array sizes logic?
        # Or hardcode for now based on "Hybrid" definition.

        # Heuristic: num_rx_ports = rx_array.num_ant (if fully digital) or specific count.
        # For HBF MVP, let's assume we reduce 4x4 panel (16 el) to 4 ports?
        num_tx_ports = 4
        num_rx_ports = 1  # UT

        if (
            self.hybrid_channel.resource_grid.num_tx == 1
        ):  # Uplink usually (1 TX per UT)
            # Logic fix: SLS simulaiton sets num_tx based on direction.
            pass

        # Call hybrid generation
        # We need batch_size
        return self.hybrid_channel.get_port_channel(
            self.batch_size, num_tx_ports=num_tx_ports, num_rx_ports=num_rx_ports
        )


class HybridSystemLevelSimulator(SystemLevelSimulator):
    """
    Extensions for Hybrid Beamforming.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # After super init, we swap the channel generation logic if needed
        # But we actually override call(), so we just need to ensure setup is correct.

        # We need to instantiate HybridOFDMChannel
        # It needs tx_array, rx_array.
        # self.ut_array, self.bs_array are available.

        if self.direction == "uplink":
            self.hybrid_channel = HybridOFDMChannel(
                self.channel_model, self.resource_grid, self.ut_array, self.bs_array
            )
        else:
            self.hybrid_channel = HybridOFDMChannel(
                self.channel_model, self.resource_grid, self.bs_array, self.ut_array
            )

    @tf.function(jit_compile=True)  # Commented out JIT for debugging
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
        # Mostly a copy of SystemLevelSimulator.call but using HybridChannelMatrix

        # -------------- #
        # Initialization #
        # -------------- #
        hist = init_result_history(
            self.batch_size, num_slots, self.num_bs, self.num_ut_per_sector
        )

        last_harq_feedback, sinr_eff_feedback, num_decoded_bits = self._reset(
            bler_target, olla_delta_up
        )

        # Initialize Hybrid Channel Matrix
        # Note: We pass num_tx (Users) and num_rx (BSs) for fading calc
        self.channel_matrix = HybridChannelMatrix(
            self.hybrid_channel,
            self.batch_size,
            self.num_rx,
            self.num_tx,  # Users count
            self.coherence_time,
            precision=self.precision,
        )

        # Initial Channel Generation
        h_freq = self.channel_matrix()

        # --------------- #
        # Simulate a slot #
        # --------------- #
        # Copying simulate_slot logic...
        # Since we cannot easily inherit the inner function, we strictly copy-paste the loop structure.

        def simulate_slot(
            slot, hist, harq_feedback, sinr_eff_feedback, num_decoded_bits, h_freq
        ):
            try:
                # Update Channel
                h_freq = self.channel_matrix.update(
                    None, h_freq, slot
                )  # Pass None for channel_model
                h_freq_fading = self.channel_matrix.apply_fading(h_freq)

                # --- The rest is identical to Base SLS ---
                # Scheduler
                rate_achievable_est = estimate_achievable_rate(
                    self.olla.sinr_eff_db_last,
                    self.resource_grid.num_ofdm_symbols,
                    self.resource_grid.fft_size,
                )
                is_scheduled = self.scheduler(num_decoded_bits, rate_achievable_est)

                num_allocated_sc = tf.minimum(
                    tf.reduce_sum(tf.cast(is_scheduled, tf.int32), axis=-1), 1
                )
                num_allocated_sc = tf.reduce_sum(num_allocated_sc, axis=-2)
                num_allocated_re = tf.reduce_sum(
                    tf.cast(is_scheduled, tf.int32), axis=[-1, -3, -4]
                )

                # Power Control
                pathloss_all_pairs, pathloss_serving_cell = get_pathloss(
                    h_freq_fading,
                    rx_tx_association=tf.convert_to_tensor(
                        self.stream_management.rx_tx_association
                    ),
                )
                pathloss_serving_cell = self._group_by_sector(pathloss_serving_cell)

                if self.direction == "uplink":
                    tx_power_per_ut = open_loop_uplink_power_control(
                        pathloss_serving_cell,
                        num_allocated_sc,
                        alpha=alpha_ul,
                        p0_dbm=p0_dbm_ul,
                        ut_max_power_dbm=self.ut_max_power_dbm,
                    )
                else:
                    # DL Power Control (Simplified copy)
                    one = tf.cast(1, pathloss_serving_cell.dtype)
                    rx_power_tot = tf.reduce_sum(one / pathloss_all_pairs, axis=-2)
                    rx_power_tot = self._group_by_sector(rx_power_tot)
                    interference_dl = rx_power_tot - one / pathloss_serving_cell
                    interference_dl *= dbm_to_watt(self.bs_max_power_dbm)
                    tx_power_per_ut, _ = downlink_fair_power_control(
                        pathloss_serving_cell,
                        interference_dl + self.no,
                        num_allocated_sc,
                        bs_max_power_dbm=self.bs_max_power_dbm,
                        guaranteed_power_ratio=guaranteed_power_ratio_dl,
                        fairness=fairness_dl,
                        precision=self.precision,
                    )

                tx_power = spread_across_subcarriers(
                    tx_power_per_ut,
                    is_scheduled,
                    num_tx=self.num_tx_per_sector,
                    precision=self.precision,
                )

                # PER-STREAM SINR
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

                # Link Adaptation & PHY Abstraction
                mcs_index = self.olla(
                    num_allocated_re,
                    harq_feedback=harq_feedback,
                    sinr_eff=sinr_eff_feedback,
                )
                num_decoded_bits, harq_feedback, sinr_eff, _, _ = self.phy_abs(
                    mcs_index,
                    sinr=sinr,
                    mcs_table_index=mcs_table_index,
                    mcs_category=int(self.direction == "downlink"),
                )

                sinr_eff_feedback = tf.where(
                    num_allocated_re > 0, sinr_eff, tf.cast(0.0, self.rdtype)
                )

                # Record
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
                print(f"SINR error at slot {slot}: {e}")
                hist = record_results(
                    hist,
                    slot,
                    shape=[self.batch_size, self.num_bs, self.num_ut_per_sector],
                    sim_failed=True,
                )

            # User Mobility
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

            return [
                slot + 1,
                hist,
                harq_feedback,
                sinr_eff_feedback,
                num_decoded_bits,
                h_freq,
            ]

        # Loop
        _, hist, *_ = tf.while_loop(
            lambda i, *_: i < num_slots,
            simulate_slot,
            [0, hist, last_harq_feedback, sinr_eff_feedback, num_decoded_bits, h_freq],
        )

        for key in hist:
            hist[key] = hist[key].stack()
        return hist
