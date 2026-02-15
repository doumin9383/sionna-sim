import tensorflow as tf
import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath("src"))

from wsim.common.phy.mcs import decode_mcs_index, get_mcs_table
from wsim.sls.link_adaptation import MCSLinkAdaptation


def test_mcs_logic():
    print("Running MCS Logic Tests...")

    # Test 1: decode_mcs_index (Table 1)
    m, r, s = decode_mcs_index(0, table_index=1, is_pusch=True)
    print(f"Table 1 MCS 0: mod={m}, rate={r:.4f}, sinr={s}")
    assert m == 2
    assert abs(r - 120 / 1024) < 1e-6
    assert s == -6.0

    # Test 2: decode_mcs_index (PUSCH TP, q=1)
    m, r, s = decode_mcs_index(
        0, table_index=1, is_pusch=True, transform_precoding=True, pi2bpsk=True
    )
    print(f"PUSCH TP MCS 0 (pi2bpsk): mod={m}, rate={r:.4f}, sinr={s}")
    assert m == 1
    assert abs(r - 240 / 1024) < 1e-6
    assert s == -9.0

    # Test 3: MCSLinkAdaptation initialization
    la = MCSLinkAdaptation(table_index=1, is_pusch=True)
    print(f"LA MCS Table length: {len(la.mcs_table)}")
    assert len(la.mcs_table) == 29
    assert la.mcs_table[0] == [0, 2, 120, -6.0]

    # Test 4: select_mcs
    mcs, mod, rate = la.select_mcs(15.5)
    print(f"Select MCS for 15.5dB: mcs={mcs}, mod={mod}, rate={rate:.4f}")
    # Table 1: index 12 corresponds to 15.0dB, 13 to 16.0dB.
    assert mcs == 12

    # Test 5: vectorized throughput
    sinr_tensor = tf.constant([[-10.0, 0.0, 15.5, 35.0]], dtype=tf.float32)
    thr, sel_mcs = la.get_throughput_vectorized(sinr_tensor)
    print(f"Vectorized Selected MCS: {sel_mcs.numpy()}")
    # -10 -> 0 (Actually -6.0 is min, so below that it selects index 0 with se masked? No, masked se is 0)
    # Wait, in get_throughput_vectorized: se_masked = tf.where(supported, se, 0)
    # If SINR < -6.0, no MCS is supported, so throughput should be 0 and mcs might be -1.
    assert sel_mcs.numpy()[0, 0] == -1
    assert sel_mcs.numpy()[0, 1] == 3
    assert sel_mcs.numpy()[0, 2] == 12
    assert sel_mcs.numpy()[0, 3] == 28

    print("All tests passed!")


if __name__ == "__main__":
    try:
        test_mcs_logic()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
