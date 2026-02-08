import tensorflow as tf
import numpy as np
from sionna.phy.nr import calculate_tb_size
from experiments.hybrid_beamforming.shared.tables import NR_MCS_TABLE_1


class MCSLinkAdaptation:
    """
    Link Adaptation based on 5G NR MCS Table.
    Selects MCS based on effective SINR and calculates Transport Block Size (TBS).
    """

    def __init__(self, target_bler=0.1):
        self.target_bler = target_bler
        # Table 5.1.3.1-1: MCS index table 1 for PUSCH
        # Columns: [MCS Index, Modulation Order, Target Code Rate (x1024), Required SINR (dB)]
        # Approximate required SINR for BLER=10%
        self.mcs_table = NR_MCS_TABLE_1

    def select_mcs(self, sinr_db):
        """
        Selects the highest MCS index that satisfies the SINR requirement.
        Scalar version (Python loop).
        """
        if tf.is_tensor(sinr_db):
            sinr_db = sinr_db.numpy()

        selected_mcs = 0
        modulation_order = 2
        code_rate = 120 / 1024.0

        for entry in self.mcs_table:
            mcs_idx, mod, rate_1024, req_sinr = entry
            if sinr_db >= req_sinr:
                selected_mcs = mcs_idx
                modulation_order = mod
                code_rate = rate_1024 / 1024.0
            else:
                break

        return selected_mcs, modulation_order, code_rate

    def calculate_throughput(
        self, mcs_index, num_rbs, num_layers=1, slot_duration=1e-3
    ):
        """
        Calculates throughput based on MCS index and resource allocation.
        """
        try:
            mod_order = 2
            code_rate = 0.1
            for entry in self.mcs_table:
                if entry[0] == mcs_index:
                    mod_order = entry[1]
                    code_rate = entry[2] / 1024.0
                    break

            tbs = calculate_tb_size(
                mod_order=mod_order,
                target_code_rate=code_rate,
                num_layers=num_layers,
                num_prb=num_rbs,
            )

            return (tbs / slot_duration) * (1 - self.target_bler)

        except Exception as e:
            print(f"Error calculating throughput: {e}")
            return 0.0

    def get_throughput_vectorized(self, sinr_db):
        """
        Vectorized lookup of throughput based on SINR.
        Compatible with TensorFlow tensors.

        Args:
            sinr_db (tf.Tensor): Effective SINR in dB.

        Returns:
            throughput (tf.Tensor): Throughput per Resource Element (bits/RE).
            selected_mcs (tf.Tensor): Selected MCS Index.
        """
        # Convert table to constants
        # req_sinr: [NumMCS]
        req_sinr = tf.constant([row[3] for row in self.mcs_table], dtype=sinr_db.dtype)

        # Spectral Efficiency = ModOrder * CodeRate
        # se: [NumMCS]
        se = tf.constant(
            [row[1] * (row[2] / 1024.0) for row in self.mcs_table], dtype=sinr_db.dtype
        )

        # MCS Indices
        mcs_indices = tf.range(len(self.mcs_table), dtype=tf.int32)

        # Expand dims for broadcasting
        # sinr_db shape: [..., 1] or [..., M]
        sinr_expanded = tf.expand_dims(sinr_db, -1)

        # Compare: sinr >= req ?
        # Result: [..., NumMCS] boolean
        supported = sinr_expanded >= req_sinr

        # Get SE for supported MCS
        # Use se where supported, else 0
        se_masked = tf.where(supported, se, tf.zeros_like(se))

        # Get indices for supported MCS
        # Use index where supported, else -1
        indices_masked = tf.where(supported, mcs_indices, -1)

        # Get Max SE over available MCS
        selected_se = tf.reduce_max(se_masked, axis=-1)

        # Get Max Index over available MCS
        selected_mcs = tf.reduce_max(indices_masked, axis=-1)

        # Throughput = SE * (1 - BLER_target)
        throughput = selected_se * (1.0 - self.target_bler)

        return throughput, selected_mcs
