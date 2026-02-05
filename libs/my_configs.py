import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class AntennaConfig:
    """アンテナ設定"""
    pattern: str = "iso"            # アンテナパターン (e.g., 'iso', '3gpp-3d')
    polarization: str = "VH"        # 偏波 (e.g., 'VH', 'V', 'H')
    tilt_angle: float = 0.0         # チルト角 [deg]
    num_rows: int = 1               # 行数
    num_cols: int = 1               # 列数
    element_spacing_row: float = 0.5 # 行間隔 [lambda]
    element_spacing_col: float = 0.5 # 列間隔 [lambda]

@dataclass
class WaveformConfig:
    """波形・変調設定 (主にLinkレベル用)"""
    modulation: str = "QAM16"       # 変調方式
    coderate: float = 0.5           # 符号化率
    subcarrier_spacing: float = 30e3 # サブキャリア間隔 [Hz]
    num_ofdm_symbols: int = 14      # OFDMシンボル数

@dataclass
class SimulationParameters:
    """シミュレーション共通物理パラメータ"""
    carrier_frequency: float = 3.5e9 # キャリア周波数 [Hz]
    bandwidth: float = 100e6         # 帯域幅 [Hz]
    noise_power: float = -174.0      # ノイズ電力密度 [dBm/Hz] (または絶対値)

@dataclass
class MasterConfig:
    """
    全ての実験設定を統括するクラス
    実験カテゴリとRunning Nameを持ち、結果出力パスの解決に使われる
    """
    # [管理用メタデータ]
    exp_category: str = "default"    # 実験カテゴリ (フォルダ名に対応)
    run_name: str = "run001"         # 実行名 (結果フォルダ名に対応)

    # [物理層/システム設定]
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    params: SimulationParameters = field(default_factory=SimulationParameters)

    # [その他拡張用]
    extra_params: Dict[str, Any] = field(default_factory=dict)
