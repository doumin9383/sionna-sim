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

    def get_mpr(self, waveform, rank):
        """
        Returns MPR in dB for the given waveform and rank.
        Supports both scalar and tensor inputs for rank.
        """
        if not self.mpr_table:
            return tf.zeros_like(rank, dtype=tf.float32) if tf.is_tensor(rank) else 0.0

        def lookup_mpr(r):
            rank_str = str(int(r))
            for row in self.mpr_table:
                if row.get("waveform") == waveform and row.get("rank") == rank_str:
                    return float(row.get("mpr_db", 0.0))
            return 0.0

        if tf.is_tensor(rank):
            # Use tf.map_fn or simply vectorize the lookup if table is small
            # For simplicity with the existing CSV-based table, we use tf.py_function
            mpr = tf.py_function(
                func=lambda r: np.vectorize(lookup_mpr)(r.numpy()),
                inp=[rank],
                Tout=tf.float32,
            )
            mpr.set_shape(rank.shape)
            return mpr
        else:
            return lookup_mpr(rank)
