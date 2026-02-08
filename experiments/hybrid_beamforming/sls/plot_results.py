import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend
import matplotlib.pyplot as plt
import csv
import os
import ast
import numpy as np


def plot_results(
    csv_path="simulation_results.csv", output_path="simulation_results.png"
):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    slots = []
    throughputs = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slots.append(int(row["Slot"]))
                # The CSV might contain a string representation of a numpy array like "[33.0 ...]"
                val_str = row["Average_Throughput_bps"]
                # Clean up string to be parsed
                val_str = val_str.replace("[", "").replace("]", "").strip()
                # Split by whitespace
                vals = [float(x) for x in val_str.split()]
                avg_val = np.mean(vals)
                throughputs.append(avg_val / 1e6)  # Convert to Mbps
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(slots, throughputs, marker="o", linestyle="-", color="b")
    plt.title("Average Network Throughput per Slot (SLS Verification)")
    plt.xlabel("Slot Index")
    plt.ylabel("Throughput (Mbps)")
    plt.grid(True)
    plt.ylim(bottom=0)

    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_results()
