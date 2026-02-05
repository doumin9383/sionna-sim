import sys
import os
# プロジェクトルートへのパス解決 (環境に合わせて調整が必要になる場合あり)
sys.path.append(os.path.abspath('../../../..'))

from libs.my_configs import MasterConfig, AntennaConfig, WaveformConfig, SimulationParameters

# Reconstructed Configuration
config = MasterConfig(exp_category='exp01_basic_link', run_name='case_64qam', antenna=AntennaConfig(pattern='iso', polarization='VH', tilt_angle=0.0, num_rows=1, num_cols=1, element_spacing_row=0.5, element_spacing_col=0.5), waveform=WaveformConfig(modulation='QAM64', coderate=0.5, subcarrier_spacing=30000.0, num_ofdm_symbols=14), params=SimulationParameters(carrier_frequency=3500000000.0, bandwidth=100000000.0, noise_power=-174.0), extra_params={})
