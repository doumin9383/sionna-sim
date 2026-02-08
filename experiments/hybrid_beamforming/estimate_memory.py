import numpy as np


def estimate_vram(
    batch_size,
    num_bs,
    num_ut,
    num_tx_ant,
    num_rx_ant,
    num_ofdm_symbols,
    num_subcarriers,
    precision="complex64",
):
    # Size of one complex number in bytes
    if precision == "complex64":
        bytes_per_elem = 8
    elif precision == "complex128":
        bytes_per_elem = 16
    else:
        raise ValueError("Unknown precision")

    # Dimensions: [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]
    # In Sionna: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, num_sc]
    # Note: HybridOFDMChannel returns port domain: [batch, num_rx, num_tx, num_ofdm, num_sc, num_rx_ports, num_tx_ports]
    # But usually physical channel is [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, num_sc]

    # Let's estimate the physical channel size first (Sionna internal)
    # shape: [batch, num_ut, num_rx_ant, num_bs, num_tx_ant, num_ofdm, num_sc]
    num_elements = (
        batch_size
        * num_ut
        * num_rx_ant
        * num_bs
        * num_tx_ant
        * num_ofdm_symbols
        * num_subcarriers
    )
    total_bytes = num_elements * bytes_per_elem

    print(f"--- Configuration ---")
    print(f"Batch Size: {batch_size}")
    print(f"Num BS: {num_bs} (Cells/Sectors)")
    print(f"Num UT: {num_ut}")
    print(f"BS Antennas: {num_tx_ant}")
    print(f"UT Antennas: {num_rx_ant}")
    print(f"OFDM Symbols: {num_ofdm_symbols}")
    print(f"Subcarriers: {num_subcarriers}")
    print(f"Precision: {precision}")

    print(f"--- Estimated Memory ---")
    print(f"Total Elements: {num_elements:.2e}")
    print(f"Size: {total_bytes / 1024**3:.2f} GB")
    print(f"Size: {total_bytes / 1024**4:.2f} TB")


print("=== Case 1: Minimum Setup (1 Cell, 3 Sectors, 1 UT/Sector) ===")
# 3 Sectors, 3 UTs
# 100MHz (3276 sc), 64 Tx, 4 Rx, 14 Sym
estimate_vram(1, 3, 3, 64, 4, 14, 3276)

print("\n=== Case 2: Small Cluster (7 Cells, 21 Sectors, 1 UT/Sector) ===")
# 21 Sectors, 21 UTs
estimate_vram(1, 21, 21, 64, 4, 14, 3276)

print(
    "\n=== Case 3: Reduced Subcarriers (RBG Granularity, e.g., 4 RB = 48 sc -> 273/4 = 69 RBGs) ==="
)
# 21 Sectors, 21 UTs, 69 freq points
estimate_vram(1, 21, 21, 64, 4, 14, 69)

print("\n=== Case 4: User's Massive Setup (Max 900 Sectors, Uplink) ===")
# Max 300 sites * 3 sectors = 900 Sectors.
# Assumption: 1 UE per sector -> 900 UEs.
# Rx (BS): 256 elements (Massive MIMO).
# Tx (UE): 4 elements.
# Freq: 11 RBGs (User specified '11 RRBG').
# OFDM Symbols: 14.
# Note: Uplink channel is Reciprocal of Downlink or explicitly generated.
# Sionna usually generates H: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, ...]
# In Uplink: Rx=BS(256), Tx=UE(4).
estimate_vram(1, 900, 900, 4, 256, 14, 11)

print("\n=== Case 5: LLS (PAPR/Throughput) - Single Link, Full Band ===")
# LLS is usually 1 BS, 1 UE.
# 1 BS (64 ports), 1 UE (4 ports).
# Full Bandwidth (3276 sc) for PAPR.
# Batch size might be larger (e.g. 100 or 1000) for error rate curves.
estimate_vram(100, 1, 1, 64, 4, 14, 3276)
