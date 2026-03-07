import os
import csv
import tensorflow as tf
import numpy as np


class MPRModel:
    """
    MPR (Maximum Power Reduction) Model.
    Loads MPR table from a CSV file and provides MPR values based on waveform and rank.
    """

    def __init__(self, config=None, csv_path=None):
        self.config = config
        if csv_path:
            self.csv_path = csv_path
        elif config:
            self.csv_path = config.mpr_table_path
        else:
            self.csv_path = "mpr_table.csv"

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

    def _calculate_mpr_value(self, row):
        """Calculates MPR based on the configured model type."""
        model_type = self.config.mpr_model_type if self.config else "linear"

        if model_type == "table":
            target_col = self.config.mpr_table_mode_column if self.config else "cm_db"
            val = row.get(target_col)
            # Fallback chain if target column missing
            if val is None or val == "":
                val = (
                    row.get("mpr_db")
                    or row.get("papr_db_10e-3")
                    or row.get("papr_db_99.9")
                )
            return float(val) if val is not None else 0.0

        else:  # "linear"
            # MPR = k * (PAPR - PAPR_ref) + MPR_ref
            papr = float(row.get("papr_db_10e-3") or row.get("papr_db_99.9") or 0.0)

            if self.config:
                k = self.config.mpr_linear_slope
                papr_ref = self.config.mpr_linear_ref_papr
                mpr_ref = self.config.mpr_linear_ref_backoff
            else:
                k = 0.5
                papr_ref = 9.6
                mpr_ref = 0.5

            mpr = k * (papr - papr_ref) + mpr_ref
            return max(mpr, 0.0)

    def get_mpr(self, waveform, rank, modulation, num_rb, granularity):
        """
        Returns MPR in dB for the given parameters.
        Supports both scalar and tensor inputs for rank and other parameters if needed,
        but typically called with tensors for rank and scalars/broadcasted values for others.
        """
        if not self.mpr_table:
            return tf.zeros_like(rank, dtype=tf.float32) if tf.is_tensor(rank) else 0.0

        def lookup_mpr(r, mod, rb, gran):
            if isinstance(mod, bytes):
                mod = mod.decode("utf-8")
            if isinstance(gran, bytes):
                gran = gran.decode("utf-8")

            rank_str = str(int(r))
            rb_str = str(int(rb))

            for row in self.mpr_table:
                if (
                    row.get("waveform") == waveform
                    and row.get("rank") == rank_str
                    and row.get("modulation") == mod
                    and row.get("num_rb") == rb_str
                    and row.get("granularity") == gran
                ):
                    # Found matching row
                    return self._calculate_mpr_value(row)

            # Not found: Warning and Return 0.0
            # For robustness in large sweeps, maybe print warning and return 0?
            # But missing MPR is critical.
            # print(
            #     f"Warning: MPR not found for: Waveform={waveform}, Rank={rank_str}, "
            #     f"TP={tp_str}, Mod={mod}, RB={rb_str}, Gran={gran}. Using 0.0."
            # )
            return 0.0

        # Vectorized lookup for tensors
        def vectorized_lookup(r_v, mod_v, rb_v, gran_v):
            # Convert tensors to numpy if they aren't already (py_function handles this)
            return np.vectorize(lookup_mpr)(r_v, mod_v, rb_v, gran_v)

        if tf.is_tensor(rank):
            # Ensure all inputs are tensors for py_function
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
                inp=[rank, mod_t, rb_t, gran_t],
                Tout=tf.float32,
            )
            mpr.set_shape(rank.shape)
            return mpr
        else:
            return lookup_mpr(rank, modulation, num_rb, granularity)
