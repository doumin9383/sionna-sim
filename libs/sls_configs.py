import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class AntennaConfig:
    """アンテナ設定 (SLS用)"""
    pattern: str = "iso"            # アンテナパターン (e.g., 'iso', '3gpp-3d')
    polarization: str = "VH"        # 偏波 (e.g., 'VH', 'V', 'H')
    tilt_angle: float = 0.0         # チルト角 [deg]
    num_rows: int = 1               # 行数
    num_cols: int = 1               # 列数
    element_spacing_row: float = 0.5 # 行間隔 [lambda]
    element_spacing_col: float = 0.5 # 列間隔 [lambda]

@dataclass
class CarrierConfig:
    """5G NR キャリア設定"""
    subcarrier_spacing: float = 30e3    # サブキャリア間隔 [Hz]
    fft_size: int = 4096                # FFTサイズ (Resource Grid用)
    num_ofdm_symbols: int = 14          # スロットあたりのシンボル数
    bandwidth: float = 100e6            # システム帯域幅 [Hz]
    carrier_frequency: float = 3.5e9    # キャリア周波数 [Hz]

@dataclass
class TopologyConfig:
    """トポロジー設定 (BS/UE配置)"""
    inter_site_distance: float = 500.0  # 基地局間距離 [m]
    num_sectors: int = 3                # 1サイトあたりのセクター数
    num_sites: int = 7                  # サイト数 (総BS数 = num_sites * num_sectors)
    num_ues_per_sector: int = 10        # 1セクターあたりのUE数 (合計UE数はこれに基づいて計算)
    ue_height: float = 1.5              # UEの高さ [m]
    bs_height: float = 25.0             # BSの高さ [m]
    min_dist: float = 10.0              # BS-UE間の最小距離 [m]

@dataclass
class ChannelConfig:
    """チャネルモデル設定 (3GPP)"""
    model_name: str = "UMa"             # モデル名: 'UMa', 'UMi', 'RMa'
    los_probability: float = 1.0        # LoS確率 (1.0で強制LoS, 0でNLoS, またはNoneでモデル依存)
    channel_condition: str = "LoS"      # 'LoS' or 'NLoS' (簡易設定用)

@dataclass
class TrafficConfig:
    """トラフィック設定"""
    buffer_size: int = 1024 * 1024      # バッファサイズ [bits]
    arrival_rate: float = 1e6           # 平均到着率 [bits/s]

@dataclass
class SLSMasterConfig:
    """
    SLS用のマスター設定
    (libs.my_configs.MasterConfigとは独立して定義)
    """
    # [管理用メタデータ]
    exp_category: str = "default"    # 実験カテゴリ
    run_name: str = "run001"         # 実行名

    # SLS固有の設定
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)

    # 共通設定
    antenna: AntennaConfig = field(default_factory=AntennaConfig)

    # 拡張用
    extra_params: Dict[str, Any] = field(default_factory=dict)
