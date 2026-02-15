import numpy as np
from typing import Tuple, Dict, Any, Optional

# TS 38.214 Tables
# Format: {table_index: {mcs_index: (mod_order, target_rate_x_1024, approx_sinr_db)}}
# SINR is approximate for BLER=10%.

# PDSCH MCS Tables (also used for PUSCH with CP-OFDM)
_PDSCH_TABLES = {
    1: {  # Table 5.1.3.1-1
        0: (2, 120, -6.0),
        1: (2, 157, -4.0),
        2: (2, 193, -2.0),
        3: (2, 251, 0.0),
        4: (2, 308, 2.0),
        5: (2, 379, 4.0),
        6: (2, 449, 6.0),
        7: (2, 526, 8.0),
        8: (2, 602, 10.0),
        9: (2, 679, 12.0),
        10: (4, 340, 13.0),
        11: (4, 378, 14.0),
        12: (4, 434, 15.0),
        13: (4, 490, 16.0),
        14: (4, 553, 17.0),
        15: (4, 616, 18.0),
        16: (4, 658, 19.0),
        17: (6, 438, 19.5),
        18: (6, 466, 20.0),
        19: (6, 517, 21.0),
        20: (6, 567, 22.0),
        21: (6, 616, 23.0),
        22: (6, 666, 24.0),
        23: (6, 719, 25.0),
        24: (6, 772, 26.0),
        25: (6, 822, 27.0),
        26: (6, 873, 28.0),
        27: (6, 910, 29.0),
        28: (6, 948, 30.0),
    },
    2: {  # Table 5.1.3.1-2 (256QAM)
        0: (2, 120, -6.0),
        1: (2, 193, -2.0),
        2: (2, 308, 2.0),
        3: (2, 449, 6.0),
        4: (2, 602, 10.0),
        5: (4, 378, 14.0),
        6: (4, 434, 15.0),
        7: (4, 490, 16.0),
        8: (4, 553, 17.0),
        9: (4, 616, 18.0),
        10: (4, 658, 19.0),
        11: (6, 466, 20.0),
        12: (6, 517, 21.0),
        13: (6, 567, 22.0),
        14: (6, 616, 23.0),
        15: (6, 666, 24.0),
        16: (6, 719, 25.0),
        17: (6, 772, 26.0),
        18: (6, 822, 27.0),
        19: (6, 873, 28.0),
        20: (8, 682.5, 30.0),
        21: (8, 711, 31.0),
        22: (8, 754, 32.0),
        23: (8, 797, 33.0),
        24: (8, 841, 34.0),
        25: (8, 885, 35.0),
        26: (8, 916.5, 36.5),
        27: (8, 948, 38.0),
    },
    3: {  # Table 5.1.3.1-3 (Low SE)
        0: (2, 30, -15.0),
        1: (2, 40, -13.0),
        2: (2, 50, -11.0),
        3: (2, 64, -9.0),
        4: (2, 78, -7.5),
        5: (2, 99, -6.5),
        6: (2, 120, -6.0),
        7: (2, 157, -4.0),
        8: (2, 193, -2.0),
        9: (2, 251, 0.0),
        10: (2, 308, 2.0),
        11: (2, 379, 4.0),
        12: (2, 449, 6.0),
        13: (2, 526, 8.0),
        14: (2, 602, 10.0),
        15: (4, 340, 13.0),
        16: (4, 378, 14.0),
        17: (4, 434, 15.0),
        18: (4, 490, 16.0),
        19: (4, 553, 17.0),
        20: (4, 616, 18.0),
        21: (6, 438, 19.5),
        22: (6, 466, 20.0),
        23: (6, 517, 21.0),
        24: (6, 567, 22.0),
        25: (6, 616, 23.0),
        26: (6, 666, 24.0),
        27: (6, 719, 25.0),
        28: (6, 772, 26.0),
    },
    4: {  # Table 5.1.3.1-4 (1024QAM)
        0: (2, 120, -6.0),
        1: (2, 193, -2.0),
        2: (2, 449, 6.0),
        3: (4, 378, 14.0),
        4: (4, 490, 16.0),
        5: (4, 616, 18.0),
        6: (6, 466, 20.0),
        7: (6, 517, 21.0),
        8: (6, 567, 22.0),
        9: (6, 616, 23.0),
        10: (6, 666, 24.0),
        11: (6, 719, 25.0),
        12: (6, 772, 26.0),
        13: (6, 822, 27.0),
        14: (6, 873, 28.0),
        15: (8, 682.5, 30.0),
        16: (8, 711, 31.0),
        17: (8, 754, 32.0),
        18: (8, 797, 33.0),
        19: (8, 841, 34.0),
        20: (8, 885, 35.0),
        21: (8, 916.5, 36.5),
        22: (8, 948, 38.0),
        23: (10, 805.5, 40.0),
        24: (10, 853, 42.0),
        25: (10, 900.5, 44.0),
        26: (10, 948, 46.0),
    },
}

# PUSCH MCS Tables with Transform Precoding
# Table 6.1.4.1-1
_PUSCH_TP_TABLE_1_BASE = {
    2: (2, 193, -2.0),
    3: (2, 251, 0.0),
    4: (2, 308, 2.0),
    5: (2, 379, 4.0),
    6: (2, 449, 6.0),
    7: (2, 526, 8.0),
    8: (2, 602, 10.0),
    9: (2, 679, 12.0),
    10: (4, 340, 13.0),
    11: (4, 378, 14.0),
    12: (4, 434, 15.0),
    13: (4, 490, 16.0),
    14: (4, 553, 17.0),
    15: (4, 616, 18.0),
    16: (4, 658, 19.0),
    17: (6, 466, 20.0),
    18: (6, 517, 21.0),
    19: (6, 567, 22.0),
    20: (6, 616, 23.0),
    21: (6, 666, 24.0),
    22: (6, 719, 25.0),
    23: (6, 772, 26.0),
    24: (6, 822, 27.0),
    25: (6, 873, 28.0),
    26: (6, 910, 29.0),
    27: (6, 948, 30.0),
}

# Table 6.1.4.1-2 (Low SE)
_PUSCH_TP_TABLE_2_BASE = {
    6: (2, 120, -6.0),
    7: (2, 157, -4.0),
    8: (2, 193, -2.0),
    9: (2, 251, 0.0),
    10: (2, 308, 2.0),
    11: (2, 379, 4.0),
    12: (2, 449, 6.0),
    13: (2, 526, 8.0),
    14: (2, 602, 10.0),
    15: (2, 679, 12.0),
    16: (4, 378, 14.0),
    17: (4, 434, 15.0),
    18: (4, 490, 16.0),
    19: (4, 553, 17.0),
    20: (4, 616, 18.0),
    21: (4, 658, 19.0),
    22: (4, 699, 19.5),
    23: (4, 772, 21.0),
    24: (6, 567, 22.0),
    25: (6, 616, 23.0),
    26: (6, 666, 24.0),
    27: (6, 772, 26.0),
}


def decode_mcs_index(
    mcs_index: int,
    table_index: int = 1,
    is_pusch: bool = True,
    transform_precoding: bool = False,
    pi2bpsk: bool = False,
) -> Tuple[int, float, float]:
    """
    Decodes MCS index into modulation order, target code rate, and required SINR.

    Args:
        mcs_index: MCS index (0-31)
        table_index: MCS table index (1-4)
        is_pusch: True for PUSCH, False for PDSCH
        transform_precoding: True if DFT-s-OFDM is used (PUSCH only)
        pi2bpsk: True if pi/2-BPSK is used (transform_precoding only)

    Returns:
        (modulation_order, target_code_rate, approx_sinr_db)
    """
    if not transform_precoding or not is_pusch:
        # PDSCH or CP-OFDM PUSCH
        table = _PDSCH_TABLES.get(table_index, _PDSCH_TABLES[1])
        if mcs_index not in table:
            raise ValueError(f"Invalid MCS index {mcs_index} for Table {table_index}")
        mod, rate, sinr = table[mcs_index]
        return int(mod), float(rate) / 1024.0, float(sinr)
    else:
        # PUSCH with Transform Precoding
        q = 1 if pi2bpsk else 2
        if table_index == 1:
            if mcs_index == 0:
                return q, (240.0 / q) / 1024.0, -6.0 if q == 2 else -9.0
            elif mcs_index == 1:
                return q, (314.0 / q) / 1024.0, -4.0 if q == 2 else -7.0
            elif mcs_index in _PUSCH_TP_TABLE_1_BASE:
                mod, rate, sinr = _PUSCH_TP_TABLE_1_BASE[mcs_index]
                return int(mod), float(rate) / 1024.0, float(sinr)
        elif table_index == 2:
            if mcs_index < 6:
                rates = [60, 80, 100, 128, 156, 198]
                sinrs = [-11.0, -10.0, -9.0, -8.0, -7.5, -6.5]  # Estimated
                return q, (rates[mcs_index] / q) / 1024.0, sinrs[mcs_index]
            elif mcs_index in _PUSCH_TP_TABLE_2_BASE:
                mod, rate, sinr = _PUSCH_TP_TABLE_2_BASE[mcs_index]
                return int(mod), float(rate) / 1024.0, float(sinr)

        raise ValueError(
            f"Invalid MCS index {mcs_index} for PUSCH TP Table {table_index}"
        )


def get_mcs_table(
    table_index: int = 1,
    is_pusch: bool = True,
    transform_precoding: bool = False,
    pi2bpsk: bool = False,
) -> list:
    """
    Returns the full MCS table as a list of [mcs_index, mod_order, rate_1024, sinr].
    Used for link adaptation initialization.
    """
    res = []
    # Maximum MCS index depends on table
    if not transform_precoding or not is_pusch:
        table = _PDSCH_TABLES.get(table_index, _PDSCH_TABLES[1])
        indices = sorted(table.keys())
    else:
        if table_index == 1:
            indices = list(range(28))
        else:
            indices = list(range(28))

    for i in indices:
        try:
            m, r, s = decode_mcs_index(
                i, table_index, is_pusch, transform_precoding, pi2bpsk
            )
            res.append([i, m, int(round(r * 1024)), s])
        except ValueError:
            continue
    return res
