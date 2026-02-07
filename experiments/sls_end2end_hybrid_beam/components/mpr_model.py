import os
import csv
import tensorflow as tf


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

        Args:
            waveform (str): "CP-OFDM" or "DFT-s-OFDM"
            rank (int): Transmission rank

        Returns:
            float: MPR value in dB
        """
        if not self.mpr_table:
            return 0.0

        # Implement lookup logic based on table structure
        # Expected columns: "waveform", "rank", "mpr_db"
        try:
            rank_str = str(rank)
            for row in self.mpr_table:
                # Check match (assuming case-insensitive for waveform might be safer, but exact for now)
                if row.get("waveform") == waveform and row.get("rank") == rank_str:
                    return float(row.get("mpr_db", 0.0))

            # Fallback if specific entry not found
            return 0.0
        except Exception as e:
            print(f"Error querying MPR table: {e}")
            return 0.0
