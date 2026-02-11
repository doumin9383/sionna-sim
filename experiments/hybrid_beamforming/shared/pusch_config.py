import numpy as np

# from .utils import generate_prng_seq
# from .config import Config
from sionna.phy.nr import PUSCHConfig as _PUSCHConfig

# from sionna.phy import nr
# from .utils import calculate_tb_size


class PUSCHConfig(_PUSCHConfig):
    def __init__(
        self, carrier_config=None, pusch_dmrs_config=None, tb_config=None, **kwargs
    ):
        super().__init__(
            carrier_config=None, pusch_dmrs_config=None, tb_config=None, **kwargs
        )
        self._external_precoding_matrix = None  # non-codebook用

    # ---precoding---#
    @property
    def precoding(self):
        """
        `str`, "non-codebook" (default), "codebook", "non-coherent" :
            PUSCH transmission scheme
        """
        self._ifndef("precoding", "non-codebook")
        return self._precoding

    @precoding.setter
    def precoding(self, value):
        # 3つのモードを許容するように変更
        valid_modes = ["codebook", "non-codebook", "non-coherent"]
        assert (
            value in valid_modes
        ), f"Unknown value for precoding. Must be one of {valid_modes}"
        self._precoding = value

    # --- Precoding Matrix Property ---
    @property
    def precoding_matrix(self):
        """モードに応じたプリコーディング行列 W を返す"""

        # 1. non-coherent: 単位行列を即座に返す（親を呼ばない）
        if self.precoding == "non-coherent":
            W = np.eye(self.num_antenna_ports, self.num_layers, dtype=np.complex64)
            return tf.constant(W)

        # 2. non-codebook: 外部注入された行列を返す（親を呼ばない）
        elif self.precoding == "non-codebook":
            if self._external_precoding_matrix is None:
                raise ValueError("Precoding matrix must be set for non-codebook mode.")
            return self._external_precoding_matrix

        # 3. codebook: 親クラス (_PUSCHConfig) の巨大なコードブック生成ロジックに丸投げ
        elif self.precoding == "codebook":
            # 親クラスのプロパティを明示的に呼び出す
            # これにより、あの膨大な Table 6.3.1.5-x シリーズが実行される
            w_from_parent = super().precoding_matrix

            if w_from_parent is None:
                return None

            # 親クラスが numpy で返してくる場合は TensorFlow テンソルに変換
            return tf.cast(w_from_parent, tf.complex64)

        # その他（フォールバック）
        return None

    @precoding_matrix.setter
    def precoding_matrix(self, value):
        """外部からプリコーディング行列を注入する（non-codebookモード用）"""
        if value is None:
            self._external_precoding_matrix = None
            return

        # 形状のバリデーション
        # 期待される形状: [num_antenna_ports, num_layers]
        expected_shape = [self.num_antenna_ports, self.num_layers]

        if tf.is_tensor(value):
            v_shape = value.shape.as_list()
        else:
            v_shape = list(np.shape(value))

        # Rankチェック
        assert (
            len(v_shape) == 2
        ), f"Precoding matrix must be a 2D matrix, but got rank {len(v_shape)}."

        # Dimensionチェック
        assert (
            v_shape == expected_shape
        ), f"Inconsistent precoding matrix shape. Expected {expected_shape}, but got {v_shape}."

        self._external_precoding_matrix = tf.cast(value, tf.complex64)

    # ---transform_precoding---#
    @property
    def transform_precoding(self):
        self._ifndef("transform_precoding", False)
        return self._transform_precoding

    @transform_precoding.setter
    def transform_precoding(self, value):
        self._transform_precoding = value
        # ぶら下がっている tb_config にも教えてあげる
        self.tb_config.transform_precoding = value

    def _check_tpmi_validity(self):
        """TPMIのバリデーション (元の check_config 内のロジックを分離)"""
        # Codebookベース: 複数アンテナからレイヤーを選択(TPMIを使用)
        if len(self.dmrs.dmrs_port_set) > 0:
            assert (
                len(self.dmrs.dmrs_port_set) == self.num_layers
            ), "num_layers must be equal to the number of dmrs ports"
        assert (
            self.num_layers <= self.num_antenna_ports
        ), "num_layers must be <= num_antenna_ports"
        assert (
            self.num_antenna_ports >= 2
        ), "codebook precoding requires two or more antenna ports"

    def check_config(self):
        """Test if the compound configuration is valid"""

        self.carrier.check_config()
        self.dmrs.check_config()
        # ------------------------------------------------------------------------------------------
        # プリコーディング方式ごとのチェック（コードブック、非コードブック、非コヒーレントに変更）
        # ------------------------------------------------------------------------------------------
        if self.precoding == "codebook":
            # codebook の時だけ TPMI の範囲チェックを走らせる
            self._check_tpmi_validity()
            assert self.num_antenna_ports >= 2, "Codebook requires >= 2 ports"

        elif self.precoding == "non-coherent":
            # 単位行列で通すため、ポート数がレイヤー数以上であればOK
            assert (
                self.num_antenna_ports >= self.num_layers
            ), "Antenna ports must be >= num_layers for non-coherent"

        elif self.precoding == "non-codebook":
            # Non-codebookベース: レイヤーごとにプリコーディングベクトルが指定される想定
            assert (
                self.num_layers == self.num_antenna_ports
            ), "For non-codebook in this context, num_layers must match num_antenna_ports (Sionna convention)"

        else:
            # Check that num_layers==num_antenna_ports
            assert (
                self.num_layers == self.num_antenna_ports
            ), "num_layers must be == num_antenna_ports"

        # ------------------------------------------------------------------------------------------
        # 以降は親のコピペ
        # ------------------------------------------------------------------------------------------

        # Check Tables 6.4.1.1.3-3/4 are valid
        if self.dmrs.length == 1:
            if self.mapping_type == "A":
                assert self.symbol_allocation[1] >= 4, "Symbol allocation is too short"
        else:
            assert (
                self.dmrs.additional_position < 2
            ), "dmrs.additional_position must be <2 for this dmrs.length"
            assert self.symbol_allocation[1] >= 4, "Symbol allocation too short"
            if self.mapping_type == "B":
                assert self.symbol_allocation[1] >= 5, "Symbol allocation is too short"

        # Check type_a and additional_position position
        if self.mapping_type == "A":
            if self.dmrs.additional_position == 3:
                assert (
                    self.dmrs.type_a_position == 2
                ), "additional_position=3 only allowed for type_a_position=2"

        # Check for valid TMPI
        # Check for valid TPMI (codebookモードの時だけ意味を持つ)
        if self.precoding == "codebook":
            if self.num_layers == 1:
                if self.num_antenna_ports == 2:
                    assert self.tpmi in range(6), "tpmi must be in [0,...,5]"
                elif self.num_antenna_ports == 4:
                    assert self.tpmi in range(28), "tpmi must be in [0,...,27]"
            elif self.num_layers == 2:
                if self.num_antenna_ports == 2:
                    assert self.tpmi in range(3), "tpmi must be in [0,...,2]"
                elif self.num_antenna_ports == 4:
                    assert self.tpmi in range(22), "tpmi must be in [0,...,21]"
            elif self.num_layers == 3:
                assert self.tpmi in range(7), "tpmi must be in [0,...,6]"
            elif self.num_layers == 4:
                assert self.tpmi in range(5), "tpmi must be in [0,...,4]"

        # Check that symbol allocation is valid
        if self.carrier.cyclic_prefix == "normal":
            max_length = 14
        else:  # cyclic_prefix == "extended"
            max_length = 12
        if self.mapping_type == "A":
            assert (
                self.symbol_allocation[0] == 0
            ), "symbol_allocation[0] must be 0 for mapping_type A"
            assert (
                4 <= self.symbol_allocation[1] <= max_length
            ), "symbol_allocation[1] must be in [4, 14 (or 12)]"
            if self.dmrs.length == 2:
                assert (
                    self.symbol_allocation[1] >= 4
                ), "symbol_allocation[1] must be >=4 for dmrs.length==2"
        elif self.mapping_type == "B":
            assert (
                0 <= self.symbol_allocation[0] <= 13
            ), "symbol_allocation[0] must be in [0,13] for mapping_type B"
            assert (
                1 <= self.symbol_allocation[1] <= max_length
            ), "symbol_allocation[1] must be in [1, 14 (or 12)]"
            if self.dmrs.length == 2:
                assert (
                    self.symbol_allocation[1] >= 5
                ), "symbol_allocation[1] must be >=5 for dmrs.length==2"
        assert (
            self.symbol_allocation[0] + self.symbol_allocation[1] <= max_length
        ), "symbol_allocation[0]+symbol_allocation[1] must be < 14 (or 12)"

        attr_list = [
            "n_size_bwp",
            "n_start_bwp",
            "num_layers",
            "mapping_type",
            "symbol_allocation",
            "n_rnti",
            "precoding",
            "transform_precoding",
            "tpmi",
        ]
        for attr in attr_list:
            value = getattr(self, attr)
            setattr(self, attr, value)

        # check that TBConfig is configured for "PUSCH"
        assert (
            self.tb.channel_type == "PUSCH"
        ), 'TB_config must be configured for "PUSCH" transmission.'

        # Check that the number of DMRS ports equals the number of layers
        # if dmrs_port_set has been set. Otherwise, this is
        # automatically ensured.
        if len(self.dmrs.dmrs_port_set) > 0:
            assert self.num_layers == len(
                self.dmrs.dmrs_port_set
            ), "num_layers must equal the number of DMRS ports"

        return True


def check_pusch_configs(pusch_configs):

    # Check that pusch_configs is a list
    assert isinstance(
        pusch_configs, list
    ), """pusch_configs must be a Sequence of instances of PUSCHConfig"""

    # Iterate through all pusch_configs and check their type and validity
    for pusch_config in pusch_configs:
        assert isinstance(
            pusch_config, PUSCHConfig
        ), """All elements of pusch_configs must be instances of PUSCHConfig"""

        pusch_config.check_config()

    # Create dictionary with extracted configuration parameters
    pc = pusch_configs[0]
    carrier = pc.carrier

    params = {
        "num_bits_per_symbol": pc.tb.num_bits_per_symbol,
        "num_tx": len(pusch_configs),
        "num_layers": pc.num_layers,
        "num_subcarriers": pc.num_subcarriers,
        "num_ofdm_symbols": pc.symbol_allocation[1],
        "subcarrier_spacing": pc.carrier.subcarrier_spacing * 1e3,
        "num_antenna_ports": pc.num_antenna_ports,
        "precoding": pc.precoding,
        "precoding_matrices": [],
        "pusch_config": pc,
        "carrier_config": pc.carrier,
        "num_coded_bits": pc.num_coded_bits,
        "target_coderate": pc.tb.target_coderate,
        "n_id": [],
        "n_rnti": [],
        "tb_size": pc.tb_size,
        "dmrs_length": pc.dmrs.length,
        "dmrs_additional_position": pc.dmrs.additional_position,
        "num_cdm_groups_without_data": pc.dmrs.num_cdm_groups_without_data,
    }
    params["bandwidth"] = params["num_subcarriers"] * params["subcarrier_spacing"]
    params["cyclic_prefix_length"] = np.ceil(
        carrier.cyclic_prefix_length * params["bandwidth"]
    )

    for pusch_config in pusch_configs:
        # codebook限定にせず、どのモードでも .precoding_matrix プロパティから回収する
        # これにより identity matrix も外部注入行列も一括で params に入る
        params["precoding_matrices"].append(pusch_config.precoding_matrix)

        # n_rnti and n_id are given per tx
        if pusch_config.tb.n_id is None:
            params["n_id"].append(pusch_config.carrier.n_cell_id)
        else:
            params["n_id"].append(pusch_config.tb.n_id)
        params["n_rnti"].append(pusch_config.n_rnti)

    params["precoding_matrices"] = tf.stack(params["precoding_matrices"], axis=0)
    return params
