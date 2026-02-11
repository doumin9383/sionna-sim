import tensorflow as tf
from sionna.phy.channel import TBConfig as _TBConfig
from .decode_mcs_table import decode_mcs_index


class TBConfig(_TBConfig):
    def __init__(self, **kwargs):
        # 親の初期化前に必要なフラグを立てておく
        self._transform_precoding = kwargs.get("transform_precoding", False)
        self._pi2bpsk = kwargs.get("pi2bpsk", False)

        super().__init__(**kwargs)

        # 値を保持するためのプライベート変数
        self._mod_order = None
        self._coderate = None

    def _update_mcs_info(self):
        """パラメータ変更時に一度だけデコードを走らせる"""
        # 毎回 .numpy() して python float に戻すのはオーバーヘッドが大きいので
        # 必要な時に一括計算してキャッシュする
        m, r = decode_mcs_index(
            self._mcs_index,
            self._mcs_table,
            is_pusch=(self._channel_type == "PUSCH"),
            transform_precoding=self._transform_precoding,
            pi2bpsk=self._pi2bpsk,
        )
        self._mod_order = int(m)
        self._coderate = float(r)

    @property
    def transform_precoding(self):
        return self._transform_precoding

    @transform_precoding.setter
    def transform_precoding(self, value):
        self._transform_precoding = bool(value)
        self._update_mcs_info()

    @property
    def mcs_index(self):
        self._ifndef("mcs_index", 14)
        return self._mcs_index

    @mcs_index.setter
    def mcs_index(self, value):
        # 3GPP Rel-15以降、テーブルによっては最大31まであるが、
        # assert value in range(32), "mcs_index must be in range from 0 to 31."
        # 元のクラスが28制限なので一旦それに合わせる
        assert value in range(29), "mcs_index must be in range from 0 to 28."
        self._mcs_index = value
        self._update_mcs_info()

    @property
    def mcs_table(self):
        self._ifndef("mcs_table", 1)
        return self._mcs_table

    @mcs_table.setter
    def mcs_table(self, value):
        assert value in range(1, 5), "mcs_table must be in range from 1 to 4"
        self._mcs_table = value
        self._update_mcs_info()

    @property
    def channel_type(self):
        self._ifndef("channel_type", "PUSCH")
        return self._channel_type

    @channel_type.setter
    def channel_type(self, value):
        assert value in ("PUSCH", "PDSCH"), 'Only "PUSCH" and "PDSCH" are supported'
        if getattr(self, "_channel_type", None) != value:
            self._channel_type = value
            self._update_mcs_info()

    # --- キャッシュされた値を返すプロパティ ---

    @property
    def target_coderate(self):
        return self._coderate

    @property
    def num_bits_per_symbol(self):
        return self._mod_order

    def check_config(self):
        """Sionna 本来の無駄な setter 発火を抑制"""
        # 1. まず MCS 情報を確定（ここが心臓部）
        self._update_mcs_info()

        # 2. 残りの、デコードに影響しない属性だけ setter を叩く
        attr_list = [
            # 記録用
            # "mcs_index",
            #  "mcs_table",
            #  "channel_type",
            "n_id"
        ]
        for attr in attr_list:
            value = getattr(self, attr)
            setattr(self, attr, value)
