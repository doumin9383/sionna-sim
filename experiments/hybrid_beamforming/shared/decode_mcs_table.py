from sionna.phy.utils import decode_mcs_index as _decode_mcs_index


def decode_mcs_index(mcs_index, table_index, channel_type, transform_precoding=False):
    """
    DFT-s-OFDM (transform precoding) を考慮して正しく MCS をデコードするラッパー
    """
    is_pusch = channel_type == "PUSCH"

    # PUSCHの場合、transform_precoding が True なら専用のテーブル(q=1系)を参照
    # transform_precoding が False なら CP-OFDM 用テーブルを参照
    m, r = _decode_mcs_index(
        mcs_index,
        table_index=table_index,
        is_pusch=is_pusch,
        transform_precoding=transform_precoding if is_pusch else False,
    )
    return int(m), float(r)
