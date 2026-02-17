import os
import csv
import tensorflow as tf
import numpy as np


class MPRModel:
    """
    MPR (Maximum Power Reduction) Model.
    Loads MPR table from a CSV file and provides MPR values based on waveform and rank.
    """

    def __init__(self, csv_path="mpr_table.csv"):
        self.csv_path = csv_path
        self.mpr_table = []
        self._load_table()

    def _load_table(self):
        """Loads the MPR table from CSV if it exists."""
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, mode="r", newline="") as f:
                    reader = csv.DictReader(f)
                    self.mpr_table = list(reader)
                print(f"Loaded MPR table from {self.csv_path}")
            except Exception as e:
                print(f"Failed to load MPR table: {e}")
                self.mpr_table = []
        else:
            print(f"MPR table not found at {self.csv_path}. Using 0.0 dB fallback.")
            self.mpr_table = []

    def get_mpr(
        self, waveform, rank, transform_precoding, modulation, num_rb, granularity
    ):
        """
        Returns MPR in dB for the given parameters.
        Supports both scalar and tensor inputs for rank and other parameters if needed,
        but typically called with tensors for rank and scalars/broadcasted values for others.
        """
        if not self.mpr_table:
            return tf.zeros_like(rank, dtype=tf.float32) if tf.is_tensor(rank) else 0.0

        # Note: In our current sls simulator, rank might be a tensor,
        # but waveform, transform_precoding, modulation, num_rb, granularity
        # are often treated as common across the batch (or at least waveform/gran/tp are).
        # We handle rank as a tensor.

        def lookup_mpr(r, tp, mod, rb, gran):
            if isinstance(mod, bytes):
                mod = mod.decode("utf-8")
            if isinstance(gran, bytes):
                gran = gran.decode("utf-8")

            rank_str = str(int(r))
            tp_str = str(bool(tp))
            rb_str = str(int(rb))

            for row in self.mpr_table:
                if (
                    row.get("waveform") == waveform
                    and row.get("rank") == rank_str
                    and row.get("transform_precoding") == tp_str
                    and row.get("modulation") == mod
                    and row.get("num_rb") == rb_str
                    and row.get("granularity") == gran
                ):
                    val = (
                        row.get("mpr_db")
                        or row.get("papr_db_99.9")
                        or row.get("papr_db_10e-3")
                    )
                    return float(val) if val is not None else 0.0

            # Not found: Raise Error
            raise ValueError(
                f"MPR not found for: Waveform={waveform}, Rank={rank_str}, "
                f"TP={tp_str}, Mod={mod}, RB={rb_str}, Gran={gran}"
            )

        # Vectorized lookup for tensors
        # We assume for now that waveform and granularity are constant for the call
        # but rank, tp, mod, rb could potentially vary (though sls often fixes rb/tp per drop)

        def vectorized_lookup(r_v, tp_v, mod_v, rb_v, gran_v):
            # Convert tensors to numpy if they aren't already (py_function handles this)
            return np.vectorize(lookup_mpr)(r_v, tp_v, mod_v, rb_v, gran_v)

        if tf.is_tensor(rank):
            # Ensure all inputs are tensors for py_function
            tp_t = (
                tf.convert_to_tensor(transform_precoding)
                if not tf.is_tensor(transform_precoding)
                else transform_precoding
            )
            mod_t = (
                tf.convert_to_tensor(modulation)
                if not tf.is_tensor(modulation)
                else modulation
            )
            rb_t = tf.convert_to_tensor(num_rb) if not tf.is_tensor(num_rb) else num_rb
            gran_t = (
                tf.convert_to_tensor(granularity)
                if not tf.is_tensor(granularity)
                else granularity
            )

            mpr = tf.py_function(
                func=vectorized_lookup,
                inp=[rank, tp_t, mod_t, rb_t, gran_t],
                Tout=tf.float32,
            )
            mpr.set_shape(rank.shape)
            return mpr
        else:
            return lookup_mpr(
                rank, transform_precoding, modulation, num_rb, granularity
            )
