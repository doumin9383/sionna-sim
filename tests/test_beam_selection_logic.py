import tensorflow as tf
import numpy as np


def test_scatter_logic():
    print("Testing Scatter Logic for Beam Selection...")

    # Dimensions
    Batch = 2
    Num_BS = 3
    Num_UT_Per_Sector = 1
    Num_UT = Num_BS * Num_UT_Per_Sector  # 3
    SC = 2
    RxA = 2
    TxA = 2

    # 1. Create Dummy h_serv (Channel from UT to Serving BS)
    # Shape: [Batch, Num_UT, SC, RxA, TxA]
    # We fill it with identifiable values to verify mapping
    # content = batch_idx * 100 + ut_idx
    h_serv_np = np.zeros((Batch, Num_UT, SC, RxA, TxA), dtype=np.float32)
    for b in range(Batch):
        for u in range(Num_UT):
            h_serv_np[b, u, :, :, :] = b * 100 + u

    h_serv = tf.constant(h_serv_np)

    # 2. Define Non-Sequential Serving BS IDs
    # Default sequential would be [0, 1, 2]
    # Let's mix it up:
    # Batch 0: UT0->BS2, UT1->BS0, UT2->BS1
    # Batch 1: UT0->BS1, UT1->BS2, UT2->BS0
    serving_bs_ids_np = np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int32)

    serving_bs_ids = tf.constant(serving_bs_ids_np)

    # 3. Apply Scatter Logic (Copied from Simulator)
    B = Batch
    N_BS = Num_BS
    N_UT_Sec = Num_UT_Per_Sector
    N_UT_Total = Num_UT
    FFT = SC

    batch_indices = tf.range(B)[:, None]  # [B, 1]
    batch_indices = tf.broadcast_to(batch_indices, [B, N_UT_Total])  # [B, Num_UT]

    bs_indices = tf.cast(serving_bs_ids, tf.int32)  # [B, Num_UT]

    slot_indices = tf.zeros_like(bs_indices)

    indices = tf.stack(
        [batch_indices, bs_indices, slot_indices], axis=-1
    )  # [B, Num_UT, 3]

    h_serv_flat = tf.reshape(h_serv, [B, N_UT_Total, -1])

    feature_dim = FFT * RxA * TxA
    h_bs_shape = [B, N_BS, N_UT_Sec, feature_dim]

    h_bs_flat = tf.scatter_nd(indices, h_serv_flat, h_bs_shape)

    h_bs = tf.reshape(h_bs_flat, [B, N_BS, N_UT_Sec, FFT, RxA, TxA])

    # 4. Verify Mapping
    # h_bs: [Batch, Num_BS, Num_UT_Per_Sector, SC, RxA, TxA]

    print("Verifying Batch 0...")
    # Batch 0:
    # BS 0 should have UT 1 (Value 1)
    val_bs0 = h_bs[0, 0, 0, 0, 0, 0].numpy()
    assert val_bs0 == 1.0, f"Expected 1.0, got {val_bs0}"

    # BS 1 should have UT 2 (Value 2)
    val_bs1 = h_bs[0, 1, 0, 0, 0, 0].numpy()
    assert val_bs1 == 2.0, f"Expected 2.0, got {val_bs1}"

    # BS 2 should have UT 0 (Value 0)
    val_bs2 = h_bs[0, 2, 0, 0, 0, 0].numpy()
    assert val_bs2 == 0.0, f"Expected 0.0, got {val_bs2}"

    print("Verifying Batch 1...")
    # Batch 1:
    # BS 0 should have UT 2 (Value 100 + 2 = 102)
    val_bs0_b1 = h_bs[1, 0, 0, 0, 0, 0].numpy()
    assert val_bs0_b1 == 102.0, f"Expected 102.0, got {val_bs0_b1}"

    # BS 1 should have UT 0 (Value 100 + 0 = 100)
    val_bs1_b1 = h_bs[1, 1, 0, 0, 0, 0].numpy()
    assert val_bs1_b1 == 100.0, f"Expected 100.0, got {val_bs1_b1}"

    # BS 2 should have UT 1 (Value 100 + 1 = 101)
    val_bs2_b1 = h_bs[1, 2, 0, 0, 0, 0].numpy()
    assert val_bs2_b1 == 101.0, f"Expected 101.0, got {val_bs2_b1}"

    print("Scatter Logic Verification PASSED!")


if __name__ == "__main__":
    test_scatter_logic()
