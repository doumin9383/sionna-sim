# Introduction to Iterative Detection and Decoding
In this notebook, you will learn how to set-up an iterative detection and decoding (IDD) scheme (first presented in [1]) by combining multiple available components in Sionna.

For a gentle introduction to MIMO simulations, we refer to the notebooks ["Simple MIMO Simulations"](https://nvlabs.github.io/sionna/phy/tutorials/Simple_MIMO_Simulation.html) and ["MIMO OFDM Transmissions over CDL"](https://nvlabs.github.io/sionna/phy/tutorials/MIMO_OFDM_Transmissions_over_CDL.html).

You will evaluate the performance of IDD with OFDM MIMO detection and soft-input soft-output (SISO) LDPC decoding and compare it againts several non-iterative detectors, such as soft-output LMMSE, K-Best, and expectation propagation (EP), as well as iterative SISO MMSE-PIC detection [2].

For the non-IDD models, the signal processing pipeline looks as follows:

![block_diagram.png](Introduction_to_Iterative_Detection_and_Decoding_files/block_diagram.png)

## Iterative Detection and Decoding
The IDD MIMO receiver iteratively exchanges soft-information between the data detector and the channel decoder, which works as follows:

![idd_diagram.png](Introduction_to_Iterative_Detection_and_Decoding_files/idd_diagram.png)

We denote by $\mathrm{L}^{D}$ the *a posteriori* information (represented by log-likelihood ratios, LLRs) and by $\mathrm{L}^{E} = \mathrm{L}^{D} - \mathrm{L}^{A}$ the extrinsic information, which corresponds to the information gain in $\mathrm{L}^{D}$ relative to the *a priori* information $\mathrm{L}^{A}$. The *a priori* LLRs represent soft information, provided to either the input of the detector (i.e., $\mathrm{L}^{A}_{Det}$) or the decoder (i.e., $\mathrm{L}^{A}_{Dec}$). While exchanging extrinsic information is standard for classical IDD, the SISO MMSE-PIC detector [2] turned out to work better when provided with the full *a posteriori* information from the decoder.

Originally, IDD was proposed with a resetting (Turbo) decoder [1]. However, state-of-the-art IDD with LDPC message passing decoding showed better performance with a non-resetting decoder [3], particularly for a low number of decoding iterations. Therefore, we will forward the decoder state (i.e., the check node to variable node messages) from each IDD iteration to the next.

## Table of contents
* [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
* [Simulation Parameters](#Simulation-Parameters)
* [Setting-up an End-to-end Block](#Setting-up-an-end-to-end-Block)
* [Non-IDD versus IDD Benchmarks](#Non-IDD-versus-IDD-Benchmarks)
* [Discussion-Optimizing IDD with Machine Learning](#Discussion-Optimizing-IDD-with-Machine-Learning)
* [Comments](#Comments)
* [List of References](#List-of-References)

## GPU Configuration and Imports


```python
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import sim_ber, ebnodb2no, expand_to_rank
from sionna.phy.mapping import Mapper, Constellation, BinarySource
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, \
                            LinearDetector, KBestDetector, EPDetector, \
                            RemoveNulledSubcarriers, MMSEPICDetector
from sionna.phy.channel import OFDMChannel, RayleighBlockFading, gen_single_sector_topology
from sionna.phy.channel.tr38901 import UMa, Antenna, PanelArray
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation
```

## Simulation Parameters
In the following, we set the simulation parameters. Please modify at will; adapting the batch size to your hardware setup might be beneficial.

The standard configuration implements a coded 5G inspired MU-MIMO OFDM uplink transmission over 3GPP UMa channels, with 4 single-antenna UEs, 16-QAM modulation, and a 16 element dual-polarized uniform planar antenna array (UPA) at the gNB. We implement least squares channel estimation with linear interpolation. Alternatively, we implement iid Rayleigh fading channels and perfect channel state information (CSI), which can be controlled by the model parameter `perfect_csi_rayleigh`.
As channel code, we apply a rate-matched 5G LDPC code at rate 1/2.


```python
SIMPLE_SIM = False   # reduced simulation time for simple simulation if set to True
if SIMPLE_SIM:
    batch_size = int(1e1)  # number of OFDM frames to be analyzed per batch
    num_iter = 5  # number of Monte Carlo Iterations (total number of Monte Carlo trials is num_iter*batch_size)
    num_steps = 6
    tf.config.run_functions_eagerly(True)   # run eagerly for better debugging
else:
    batch_size = int(64)  # number of OFDM frames to be analyzed per batch
    num_iter = 128  # number of Monte Carlo Iterations (total number of Monte Carlo trials is num_iter*batch_size)
    num_steps = 11

ebno_db_min_perf_csi = -10  # min EbNo value in dB for perfect csi benchmarks
ebno_db_max_perf_csi = 0
ebno_db_min_cest = -10
ebno_db_max_cest = 10


NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 12*4 # 4 PRBs
SUBCARRIER_SPACING = 30e3 # Hz
CARRIER_FREQUENCY = 3.5e9 # Hz
SPEED = 3. # m/s
num_bits_per_symbol = 4 # 16 QAM
n_ue = 4 # 4 UEs
NUM_RX_ANT = 16 # 16 BS antennas
num_pilot_symbols = 2

# The user terminals (UTs) are equipped with a single antenna
# with vertial polarization.
UT_ANTENNA = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni', # Omnidirectional antenna pattern
                     carrier_frequency=CARRIER_FREQUENCY)

# The base station is equipped with an antenna
# array of 8 cross-polarized antennas,
# resulting in a total of 16 antenna elements.
BS_ARRAY = PanelArray(num_rows_per_panel=2,
                      num_cols_per_panel=4,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901', # 3GPP 38.901 antenna pattern
                      carrier_frequency=CARRIER_FREQUENCY)

# 3GPP UMa channel model is considered
channel_model_uma = UMa(carrier_frequency=CARRIER_FREQUENCY,
                    o2i_model='low',
                    ut_array=UT_ANTENNA,
                    bs_array=BS_ARRAY,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)

channel_model_rayleigh = RayleighBlockFading(num_rx=1, num_rx_ant=NUM_RX_ANT, num_tx=n_ue, num_tx_ant=1)

constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)

rx_tx_association = np.ones([1, n_ue])
sm = StreamManagement(rx_tx_association, 1)

# Parameterize the OFDM channel
rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS, pilot_ofdm_symbol_indices = [2, 11],
                  fft_size=FFT_SIZE, num_tx=n_ue,
                  pilot_pattern = "kronecker",
                  subcarrier_spacing=SUBCARRIER_SPACING)

rg.show()
plt.show()

# Parameterize the LDPC code
R = 0.5  # rate 1/2
N = int(FFT_SIZE * (NUM_OFDM_SYMBOLS - 2) * num_bits_per_symbol)
# N = int((FFT_SIZE) * (NUM_OFDM_SYMBOLS - 2) * num_bits_per_symbol)
# code length; - 12 because of 11 guard carriers and 1 DC carrier, - 2 becaues of 2 pilot symbols
K = int(N * R)  # number of information bits per codeword

```


    
![png](Introduction_to_Iterative_Detection_and_Decoding_files/Introduction_to_Iterative_Detection_and_Decoding_9_0.png)
    


## Setting-up an End-to-end Block

Now, we define the baseline models for benchmarking. Let us start with the non-IDD models.


```python
class NonIddModel(Block):
    def __init__(self, num_bp_iter=12, detector='lmmse', cest_type="LS", interp="lin", perfect_csi_rayleigh=False):
        super().__init__()
        self._num_bp_iter = int(num_bp_iter)
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(K, N, num_bits_per_symbol=num_bits_per_symbol)
        self._mapper = Mapper(constellation=constellation)
        self._rg_mapper = ResourceGridMapper(rg)

        # Channel
        if perfect_csi_rayleigh:
            self._channel_model = channel_model_rayleigh
        else:
            self._channel_model = channel_model_uma

        self._channel = OFDMChannel(channel_model=self._channel_model,
                                    resource_grid=rg,
                                    add_awgn=True, normalize_channel=True, return_channel=True)

        # Receiver
        self._cest_type = cest_type
        self._interp = interp

        # Channel estimation
        self._perfect_csi_rayleigh = perfect_csi_rayleigh
        if self._perfect_csi_rayleigh:
            self._removeNulledSc = RemoveNulledSubcarriers(rg)
        elif cest_type == "LS":
            self._ls_est = LSChannelEstimator(rg, interpolation_type=interp)
        else:
            raise NotImplementedError('Not implemented:' + cest_type)

        # Detection
        if detector == "lmmse":
            self._detector = LinearDetector("lmmse", 'bit', "maxlog", rg, sm, constellation_type="qam",
                                            num_bits_per_symbol=num_bits_per_symbol, hard_out=False)
        elif detector == "k-best":
            k = 64
            self._detector = KBestDetector('bit', n_ue, k, rg, sm, constellation_type="qam",
                                           num_bits_per_symbol=num_bits_per_symbol, hard_out=False)
        elif detector == "ep":
            l = 10
            self._detector = EPDetector('bit', rg, sm, num_bits_per_symbol, l=l, hard_out=False)

        # Forward error correction (decoder)
        self._decoder = LDPC5GDecoder(self._encoder, return_infobits=True, hard_out=True, num_iter=num_bp_iter, cn_update='minsum')

    def new_topology(self, batch_size):
        """Set new topology"""
        if isinstance(self._channel_model, UMa):
            # sensible values according to 3GPP standard, no mobility by default
            topology = gen_single_sector_topology(batch_size,
                                                  n_ue, max_ut_velocity=SPEED,
                                                  scenario="uma")
            self._channel_model.set_topology(*topology)

    @tf.function  # We don't use jit_compile=True to ensure better numerical stability
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)

        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db=ebno_db, num_bits_per_symbol=num_bits_per_symbol,
                       coderate=R)  # normalize in OFDM freq. domain
        b = self._binary_source([batch_size, n_ue, 1, K])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel(x_rg, no_)

        ######################################
        ## Receiver
        if self._perfect_csi_rayleigh:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat),
                                    dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est(y, no)

        llr_ch = self._detector(y, h_hat, chan_est_var, no)  # detector
        b_hat = self._decoder(llr_ch)
        return b, b_hat
```

Next, we implement the IDD model with a non-resetting LDPC decoder, as in [3], i.e., we forward the LLRs and decoder state from one IDD iteration to the following.


```python
class IddModel(NonIddModel):  # inherited from NonIddModel
    def __init__(self, num_idd_iter=3, num_bp_iter_per_idd_iter=12, cest_type="LS", interp="lin", perfect_csi_rayleigh=False):
        super().__init__(num_bp_iter=num_bp_iter_per_idd_iter, detector="lmmse", cest_type=cest_type,
                         interp=interp, perfect_csi_rayleigh=perfect_csi_rayleigh)
        # first IDD detector is LMMSE as MMSE-PIC with zero-prior bils down to soft-output LMMSE
        self._num_idd_iter = num_idd_iter
        self._siso_detector = MMSEPICDetector(output="bit", resource_grid=rg, stream_management=sm,
                                              demapping_method='maxlog', constellation=constellation, num_iter=1,
                                              hard_out=False)
        self._siso_decoder = LDPC5GDecoder(self._encoder, return_infobits=False,
                                           num_iter=num_bp_iter_per_idd_iter, return_state=True, hard_out=False, cn_update='minsum')
        self._decoder = LDPC5GDecoder(self._encoder, return_infobits=True, return_state=True, hard_out=True, num_iter=num_bp_iter_per_idd_iter, cn_update='minsum')
        # last decoder must also be statefull

    @tf.function  # We don't use jit_compile=True to ensure better numerical stability
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)

        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db=ebno_db, num_bits_per_symbol=num_bits_per_symbol,
                       coderate=R)  # normalize in OFDM freq. domain
        b = self._binary_source([batch_size, n_ue, 1, K])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel(x_rg, no_)

        ######################################
        ## Receiver
        if self._perfect_csi_rayleigh:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat),
                                    dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est(y, no)

        llr_ch = self._detector(y, h_hat, chan_est_var, no)  # soft-output LMMSE detection
        msg_v2c = None

        if self._num_idd_iter >= 2:
            # perform first iteration outside the while_loop to initialize msg_v2c
            [llr_dec, msg_v2c] = self._siso_decoder(llr_ch, msg_v2c=msg_v2c)
            # forward a posteriori information from decoder

            llr_ch = self._siso_detector(y, h_hat, llr_dec, chan_est_var, no)
            # forward extrinsic information

            def idd_iter(llr_ch, msg_v2c, it):
                [llr_dec, msg_v2c] = self._siso_decoder(llr_ch, msg_v2c=msg_v2c)
                # forward a posteriori information from decoder
                llr_ch = self._siso_detector(y, h_hat, llr_dec, chan_est_var, no)
                # forward extrinsic information from detector

                it += 1
                return llr_ch, msg_v2c, it

            def idd_stop(llr_ch, msg_v2c, it):
                return tf.less(it, self._num_idd_iter - 1)

            it = tf.constant(1)     # we already performed initial detection and one full iteration
            llr_ch, msg_v2c, it = tf.while_loop(idd_stop, idd_iter, (llr_ch, msg_v2c, it), parallel_iterations=1,
                                               maximum_iterations=self._num_idd_iter - 1)
        else:
            # non-idd
            pass

        [b_hat, _] = self._decoder(llr_ch, msg_v2c=msg_v2c)    # final hard-output decoding (only returning information bits)
        return b, b_hat
```

## Non-IDD versus IDD Benchmarks


```python
# Range of SNR (dB)
snr_range_cest = np.linspace(ebno_db_min_cest, ebno_db_max_cest, num_steps)
snr_range_perf_csi = np.linspace(ebno_db_min_perf_csi, ebno_db_max_perf_csi, num_steps)

def run_idd_sim(snr_range, perfect_csi_rayleigh):
    lmmse = NonIddModel(detector="lmmse", perfect_csi_rayleigh=perfect_csi_rayleigh)
    k_best = NonIddModel(detector="k-best", perfect_csi_rayleigh=perfect_csi_rayleigh)
    ep = NonIddModel(detector="ep", perfect_csi_rayleigh=perfect_csi_rayleigh)
    idd2 = IddModel(num_idd_iter=2, perfect_csi_rayleigh=perfect_csi_rayleigh)
    idd3 = IddModel(num_idd_iter=3, perfect_csi_rayleigh=perfect_csi_rayleigh)

    ber_lmmse, bler_lmmse = sim_ber(lmmse,
                                    snr_range,
                                    batch_size=batch_size,
                                    max_mc_iter=num_iter,
                                    num_target_block_errors=int(batch_size * num_iter * 0.1))

    ber_ep, bler_ep = sim_ber(ep,
                              snr_range,
                              batch_size=batch_size,
                              max_mc_iter=num_iter,
                              num_target_block_errors=int(batch_size * num_iter * 0.1))

    ber_kbest, bler_kbest = sim_ber(k_best,
                                    snr_range,
                                    batch_size=batch_size,
                                    max_mc_iter=num_iter,
                                    num_target_block_errors=int(batch_size * num_iter * 0.1))

    ber_idd2, bler_idd2 = sim_ber(idd2,
                                  snr_range,
                                  batch_size=batch_size,
                                  max_mc_iter=num_iter,
                                  num_target_block_errors=int(batch_size * num_iter * 0.1)
                                  )

    ber_idd3, bler_idd3 = sim_ber(idd3,
                                  snr_range,
                                  batch_size=batch_size,
                                  max_mc_iter=num_iter,
                                  num_target_block_errors=int(batch_size * num_iter * 0.1))

    return bler_lmmse, bler_ep, bler_kbest, bler_idd2, bler_idd3


BLER = {}

# Perfect CSI
bler_lmmse, bler_ep, bler_kbest, bler_idd2, bler_idd3 = run_idd_sim(snr_range_perf_csi, perfect_csi_rayleigh=True)
BLER['Perf. CSI / LMMSE'] = bler_lmmse
BLER['Perf. CSI / EP'] = bler_ep
BLER['Perf. CSI / K-Best'] = bler_kbest
BLER['Perf. CSI / IDD2'] = bler_idd2
BLER['Perf. CSI / IDD3'] = bler_idd3

# Estimated CSI
bler_lmmse, bler_ep, bler_kbest, bler_idd2, bler_idd3 = run_idd_sim(snr_range_cest, perfect_csi_rayleigh=False)
BLER['Ch. Est. / LMMSE'] = bler_lmmse
BLER['Ch. Est. / EP'] = bler_ep
BLER['Ch. Est. / K-Best'] = bler_kbest
BLER['Ch. Est. / IDD2'] = bler_idd2
BLER['Ch. Est. / IDD3'] = bler_idd3
```

    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 2.1223e-01 | 1.0000e+00 |      250351 |     1179648 |         1024 |        1024 |         7.0 |reached target block errors
         -9.0 | 1.8603e-01 | 1.0000e+00 |      219449 |     1179648 |         1024 |        1024 |         0.2 |reached target block errors
         -8.0 | 1.1066e-01 | 9.7168e-01 |      130538 |     1179648 |          995 |        1024 |         0.2 |reached target block errors
         -7.0 | 1.6797e-02 | 3.4297e-01 |       49537 |     2949120 |          878 |        2560 |         0.5 |reached target block errors
         -6.0 | 1.1418e-03 | 3.1675e-02 |       34009 |    29786112 |          819 |       25856 |         5.2 |reached target block errors
         -5.0 | 8.7447e-05 | 2.2888e-03 |        3301 |    37748736 |           75 |       32768 |         6.6 |reached max iterations
         -4.0 | 2.6491e-08 | 3.0518e-05 |           1 |    37748736 |            1 |       32768 |         6.6 |reached max iterations
         -3.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |         6.6 |reached max iterations
    
    Simulation stopped as no error occurred @ EbNo = -3.0 dB.
    
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 2.1020e-01 | 1.0000e+00 |      247962 |     1179648 |         1024 |        1024 |         3.8 |reached target block errors
         -9.0 | 1.8366e-01 | 1.0000e+00 |      216660 |     1179648 |         1024 |        1024 |         0.3 |reached target block errors
         -8.0 | 9.6857e-02 | 9.5020e-01 |      114257 |     1179648 |          973 |        1024 |         0.3 |reached target block errors
         -7.0 | 1.2495e-02 | 3.0256e-01 |       40534 |     3244032 |          852 |        2816 |         0.8 |reached target block errors
         -6.0 | 4.7647e-04 | 1.7700e-02 |       17986 |    37748736 |          580 |       32768 |         9.2 |reached max iterations
         -5.0 | 3.5233e-06 | 4.2725e-04 |         133 |    37748736 |           14 |       32768 |         9.2 |reached max iterations
         -4.0 | 5.2982e-08 | 3.0518e-05 |           2 |    37748736 |            1 |       32768 |         9.2 |reached max iterations
         -3.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |         9.2 |reached max iterations
    
    Simulation stopped as no error occurred @ EbNo = -3.0 dB.
    


    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 2.1202e-01 | 1.0000e+00 |      250105 |     1179648 |         1024 |        1024 |         7.4 |reached target block errors
         -9.0 | 1.8481e-01 | 1.0000e+00 |      218011 |     1179648 |         1024 |        1024 |         3.3 |reached target block errors
         -8.0 | 1.1646e-01 | 9.9707e-01 |      137385 |     1179648 |         1021 |        1024 |         3.3 |reached target block errors
         -7.0 | 2.2363e-02 | 5.9375e-01 |       39571 |     1769472 |          912 |        1536 |         5.0 |reached target block errors
         -6.0 | 1.0629e-03 | 5.2894e-02 |       19122 |    17989632 |          826 |       15616 |        50.9 |reached target block errors
         -5.0 | 2.6385e-05 | 1.1902e-03 |         996 |    37748736 |           39 |       32768 |       106.7 |reached max iterations
         -4.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |       106.7 |reached max iterations
    
    Simulation stopped as no error occurred @ EbNo = -4.0 dB.
    
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 1.4904e-01 | 1.0000e+00 |      175813 |     1179648 |         1024 |        1024 |         6.0 |reached target block errors
         -9.0 | 9.4156e-02 | 9.9609e-01 |      111071 |     1179648 |         1020 |        1024 |         0.3 |reached target block errors
         -8.0 | 1.7746e-02 | 5.1395e-01 |       36635 |     2064384 |          921 |        1792 |         0.6 |reached target block errors
         -7.0 | 4.4343e-04 | 2.2675e-02 |       16739 |    37748736 |          743 |       32768 |        10.2 |reached max iterations
         -6.0 | 3.8412e-06 | 3.9673e-04 |         145 |    37748736 |           13 |       32768 |        10.2 |reached max iterations
         -5.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |        10.2 |reached max iterations
    
    Simulation stopped as no error occurred @ EbNo = -5.0 dB.
    
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 1.5042e-01 | 1.0000e+00 |      177440 |     1179648 |         1024 |        1024 |         5.0 |reached target block errors
         -9.0 | 9.6599e-02 | 9.9609e-01 |      113953 |     1179648 |         1020 |        1024 |         0.4 |reached target block errors
         -8.0 | 1.7984e-02 | 4.4824e-01 |       42429 |     2359296 |          918 |        2048 |         0.9 |reached target block errors
         -7.0 | 3.8775e-04 | 1.2695e-02 |       14637 |    37748736 |          416 |       32768 |        14.5 |reached max iterations
         -6.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |        14.5 |reached max iterations
    
    Simulation stopped as no error occurred @ EbNo = -6.0 dB.
    
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 2.9921e-01 | 1.0000e+00 |      352961 |     1179648 |         1024 |        1024 |        14.4 |reached target block errors
         -8.0 | 2.6382e-01 | 1.0000e+00 |      311216 |     1179648 |         1024 |        1024 |         0.3 |reached target block errors
         -6.0 | 2.2769e-01 | 1.0000e+00 |      268590 |     1179648 |         1024 |        1024 |         0.4 |reached target block errors
         -4.0 | 1.5128e-01 | 9.1797e-01 |      178461 |     1179648 |          940 |        1024 |         0.4 |reached target block errors
         -2.0 | 4.9773e-02 | 4.0186e-01 |      117429 |     2359296 |          823 |        2048 |         0.7 |reached target block errors
          0.0 | 1.1782e-02 | 1.0046e-01 |      111188 |     9437184 |          823 |        8192 |         2.9 |reached target block errors
          2.0 | 2.7987e-03 | 2.4963e-02 |      105649 |    37748736 |          818 |       32768 |        11.5 |reached max iterations
          4.0 | 1.2508e-03 | 9.2163e-03 |       47215 |    37748736 |          302 |       32768 |        11.5 |reached max iterations
          6.0 | 1.0336e-03 | 6.5308e-03 |       39017 |    37748736 |          214 |       32768 |        11.5 |reached max iterations
          8.0 | 8.7741e-04 | 5.5847e-03 |       33121 |    37748736 |          183 |       32768 |        11.4 |reached max iterations
         10.0 | 8.2003e-04 | 4.7302e-03 |       30955 |    37748736 |          155 |       32768 |        11.4 |reached max iterations
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 2.9983e-01 | 1.0000e+00 |      353698 |     1179648 |         1024 |        1024 |         7.5 |reached target block errors
         -8.0 | 2.6482e-01 | 1.0000e+00 |      312397 |     1179648 |         1024 |        1024 |         0.5 |reached target block errors
         -6.0 | 2.2826e-01 | 1.0000e+00 |      269264 |     1179648 |         1024 |        1024 |         0.4 |reached target block errors
         -4.0 | 1.4842e-01 | 9.1211e-01 |      175085 |     1179648 |          934 |        1024 |         0.4 |reached target block errors
         -2.0 | 4.1157e-02 | 3.5039e-01 |      121377 |     2949120 |          897 |        2560 |         1.1 |reached target block errors
          0.0 | 7.4100e-03 | 6.9548e-02 |      100524 |    13565952 |          819 |       11776 |         5.0 |reached target block errors
          2.0 | 2.1057e-03 | 1.6449e-02 |       79486 |    37748736 |          539 |       32768 |        14.1 |reached max iterations
          4.0 | 1.0968e-03 | 7.5684e-03 |       41402 |    37748736 |          248 |       32768 |        14.0 |reached max iterations
          6.0 | 7.8922e-04 | 5.6458e-03 |       29792 |    37748736 |          185 |       32768 |        14.0 |reached max iterations
          8.0 | 1.0145e-03 | 6.3782e-03 |       38296 |    37748736 |          209 |       32768 |        13.9 |reached max iterations
         10.0 | 8.4445e-04 | 5.8594e-03 |       31877 |    37748736 |          192 |       32768 |        13.9 |reached max iterations
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 3.0681e-01 | 1.0000e+00 |      361922 |     1179648 |         1024 |        1024 |        10.8 |reached target block errors
         -8.0 | 2.7050e-01 | 1.0000e+00 |      319091 |     1179648 |         1024 |        1024 |         3.4 |reached target block errors
         -6.0 | 2.3325e-01 | 1.0000e+00 |      275147 |     1179648 |         1024 |        1024 |         3.5 |reached target block errors
         -4.0 | 1.5728e-01 | 9.8145e-01 |      185530 |     1179648 |         1005 |        1024 |         3.5 |reached target block errors
         -2.0 | 4.7583e-02 | 4.2969e-01 |      112262 |     2359296 |          880 |        2048 |         7.0 |reached target block errors
          0.0 | 6.2228e-03 | 6.2425e-02 |       95429 |    15335424 |          831 |       13312 |        45.3 |reached target block errors
          2.0 | 1.3220e-03 | 1.0773e-02 |       49902 |    37748736 |          353 |       32768 |       111.2 |reached max iterations
          4.0 | 7.6869e-04 | 4.7302e-03 |       29017 |    37748736 |          155 |       32768 |       111.3 |reached max iterations
          6.0 | 5.9123e-04 | 3.8757e-03 |       22318 |    37748736 |          127 |       32768 |       111.1 |reached max iterations
          8.0 | 6.0389e-04 | 4.1809e-03 |       22796 |    37748736 |          137 |       32768 |       111.4 |reached max iterations
         10.0 | 8.1682e-04 | 5.1575e-03 |       30834 |    37748736 |          169 |       32768 |       111.2 |reached max iterations
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 2.7276e-01 | 1.0000e+00 |      321761 |     1179648 |         1024 |        1024 |         9.6 |reached target block errors
         -8.0 | 2.2901e-01 | 1.0000e+00 |      270153 |     1179648 |         1024 |        1024 |         0.5 |reached target block errors
         -6.0 | 1.7854e-01 | 1.0000e+00 |      210609 |     1179648 |         1024 |        1024 |         0.5 |reached target block errors
         -4.0 | 9.0361e-02 | 7.8516e-01 |      133242 |     1474560 |         1005 |        1280 |         0.6 |reached target block errors
         -2.0 | 1.8625e-02 | 1.9210e-01 |       93375 |     5013504 |          836 |        4352 |         2.0 |reached target block errors
          0.0 | 3.3976e-03 | 2.9494e-02 |      109218 |    32145408 |          823 |       27904 |        12.8 |reached target block errors
          2.0 | 1.0031e-03 | 8.1482e-03 |       37865 |    37748736 |          267 |       32768 |        15.0 |reached max iterations
          4.0 | 8.0527e-04 | 4.9744e-03 |       30398 |    37748736 |          163 |       32768 |        15.0 |reached max iterations
          6.0 | 5.6651e-04 | 3.6926e-03 |       21385 |    37748736 |          121 |       32768 |        15.0 |reached max iterations
          8.0 | 7.8596e-04 | 5.8594e-03 |       29669 |    37748736 |          192 |       32768 |        15.0 |reached max iterations
         10.0 | 7.3308e-04 | 5.7983e-03 |       27673 |    37748736 |          190 |       32768 |        15.1 |reached max iterations
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
        -10.0 | 2.7259e-01 | 1.0000e+00 |      321555 |     1179648 |         1024 |        1024 |        10.0 |reached target block errors
         -8.0 | 2.2693e-01 | 1.0000e+00 |      267699 |     1179648 |         1024 |        1024 |         0.6 |reached target block errors
         -6.0 | 1.7713e-01 | 1.0000e+00 |      208947 |     1179648 |         1024 |        1024 |         0.6 |reached target block errors
         -4.0 | 9.3064e-02 | 7.7500e-01 |      137228 |     1474560 |          992 |        1280 |         0.8 |reached target block errors
         -2.0 | 1.7267e-02 | 1.5110e-01 |      112030 |     6488064 |          851 |        5632 |         3.3 |reached target block errors
          0.0 | 3.3490e-03 | 2.4017e-02 |      126421 |    37748736 |          787 |       32768 |        19.3 |reached max iterations
          2.0 | 1.0733e-03 | 6.5613e-03 |       40517 |    37748736 |          215 |       32768 |        19.3 |reached max iterations
          4.0 | 6.9017e-04 | 3.8452e-03 |       26053 |    37748736 |          126 |       32768 |        19.3 |reached max iterations
          6.0 | 5.6526e-04 | 3.4485e-03 |       21338 |    37748736 |          113 |       32768 |        19.3 |reached max iterations
          8.0 | 6.3758e-04 | 4.3640e-03 |       24068 |    37748736 |          143 |       32768 |        19.3 |reached max iterations
         10.0 | 7.0741e-04 | 5.6763e-03 |       26704 |    37748736 |          186 |       32768 |        19.3 |reached max iterations


Finally, we plot the simulation results and observe that IDD outperforms the non-iterative methods by about 1 dB in the scenario with iid Rayleigh fading channels and perfect CSI. In the scenario with 3GPP UMa channels and estimated CSI, IDD performs slightly better than K-best, at considerably lower runtime.


```python
fig, ax = plt.subplots(1,2, figsize=(16,7))
fig.suptitle(f"{n_ue}x{NUM_RX_ANT} MU-MIMO UL | {2**num_bits_per_symbol}-QAM")

## Perfect CSI Rayleigh
ax[0].set_title("Perfect CSI iid. Rayleigh")
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / LMMSE'], 'x-', label='LMMSE', c='C0')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / EP'], 'o--', label='EP', c='C0')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / K-Best'], 's-.', label='K-Best', c='C0')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / IDD2'], 'd:', label=r'IDD $I=2$', c='C1')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / IDD3'], 'd:', label=r'IDD $I=3$', c='C2')

ax[0].set_xlabel(r"$E_b/N0$")
ax[0].set_ylabel("BLER")
ax[0].set_ylim((1e-4, 1.0))
ax[0].legend()
ax[0].grid(True)

## Estimated CSI Rayleigh
ax[1].set_title("Estimated CSI 3GPP UMa")
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / LMMSE'], 'x-', label='LMMSE', c='C0')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / EP'], 'o--', label='EP', c='C0')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / K-Best'], 's-.', label='K-Best', c='C0')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / IDD2'], 'd:', label=r'IDD $I=2$', c='C1')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / IDD3'], 'd:', label=r'IDD $I=3$', c='C2')

ax[1].set_xlabel(r"$E_b/N0$")
ax[1].set_ylabel("BLER")
ax[1].set_ylim((1e-3, 1.0))
ax[1].legend()
ax[1].grid(True)

plt.show()
```


    
![png](Introduction_to_Iterative_Detection_and_Decoding_files/Introduction_to_Iterative_Detection_and_Decoding_17_0.png)
    


## Discussion-Optimizing IDD with Machine Learning
Recent work [4] showed that IDD can be significantly improved by deep-unfolding, which applies machine learning to automatically tune hyperparameters of classical algorithms. The proposed *Deep-Unfolded Interleaved Detection and Decoding* method showed performance gains of up to 1.4 dB at the same computational complexity. A link to the simulation code is available in the ["Made with Sionna"](https://nvlabs.github.io/sionna/made_with_sionna.html#duidd-deep-unfolded-interleaved-detection-and-decoding-for-mimo-wireless-systems) section. 

## Comments

- As discussed in [3], IDD receivers with a non-resetting decoder converge faster than with resetting decoders. However, a resetting decoder (which does not forward `msg_vn`) might perform slightly better for a large number of message passing decoding iterations. Among other quantities, a scaling of the forwarded decoder state is optimized in the DUIDD receiver [4].
- With estimated channels, we observed that the MMSE-PIC output LLRs become large, much larger as with non-iterative receive processing.

## List of References

[1] B. Hochwald and S. Ten Brink, [*"Achieving near-capacity on a multiple-antenna channel,"*](https://ieeexplore.ieee.org/abstract/document/1194444) IEEE Trans. Commun., vol. 51, no. 3, pp. 389–399, Mar. 2003.

[2] C. Studer, S. Fateh, and D. Seethaler, [*"ASIC implementation of soft-input soft-output MIMO detection
using MMSE parallel interference cancellation,"*](https://ieeexplore.ieee.org/abstract/document/5779722) IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, Jul. 2011.

[3] W.-C. Sun, W.-H. Wu, C.-H. Yang, and Y.-L. Ueng, [*"An iterative detection and decoding receiver for LDPC-coded MIMO systems,"*](https://ieeexplore.ieee.org/abstract/document/7272776) IEEE Trans. Circuits Syst. I, vol. 62, no. 10, pp. 2512–2522, Oct. 2015.

[4] R. Wiesmayr, C. Dick, J. Hoydis, and C. Studer, [*"DUIDD: Deep-unfolded interleaved detection and decoding for MIMO wireless systems,"*](https://arxiv.org/abs/2212.07816) in Asilomar Conf. Signals, Syst., Comput., Oct. 2022.
