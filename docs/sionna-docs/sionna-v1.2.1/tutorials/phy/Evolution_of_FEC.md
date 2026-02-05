# From GSM to 5G - The Evolution of Forward Error Correction

This notebook compares the different FEC schemes from GSM via UMTS and LTE to 5G NR.
Please note that a *fair* comparison of different coding schemes depends on many aspects such as:

 - Decoding complexity, latency, and scalability

- Level of parallelism of the decoding algorithm and memory access patterns

- Error-floor behavior

- Rate adaptivity and flexibility

## Table of Contents
* [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
* [System Model](#System-Model)
* [Error Rate Simulations](#Error-Rate-Simulations)
* [Results for Longer Codewords](#Results-for-Longer-Codewords)

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

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation

# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource
from sionna.phy.fec.polar import Polar5GEncoder, Polar5GDecoder
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.fec.conv import ConvEncoder, ViterbiDecoder
from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder
from sionna.phy.utils import ebnodb2no, hard_decisions, PlotBER
from sionna.phy.channel import AWGN
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
```

## System Model


```python
class System_Model(Block):
    """System model for channel coding BER simulations.
    
    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. Arbitrary FEC encoder/decoder layers can be used to 
    initialize the model.
    
    Parameters
    ----------
        k: int
            number of information bits per codeword.
        
        n: int 
            codeword length.
        
        num_bits_per_symbol: int
            number of bits per QAM symbol.
            
        encoder: Sionna Block
            A Sionna Block that encodes information bit tensors.
            
        decoder: Sionna Block
            A Sionna Block that decodes llr tensors.
            
        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".
            
        sim_esno: bool  
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.
            
    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.
        
        ebno_db: float or tf.float
            A float defining the simulation SNR.
            
    Output
    ------
        (u, u_hat):
            Tuple:
        
        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.           

        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.           
    """
    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol,                 
                 encoder,
                 decoder,
                 demapping_method="app",
                 sim_esno=False):

        super().__init__()
        
        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        
        # number of bit per QAM symbol
        self.num_bits_per_symbol = num_bits_per_symbol

        # init components
        self.source = BinarySource()
       
        # initialize mapper and demapper for constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)
        
        # the channel can be replaced by more sophisticated models
        self.channel = AWGN()

        # FEC encoder / decoder
        self.encoder = encoder
        self.decoder = decoder

    @tf.function(jit_compile=True) # enable graph mode for increased throughputs
    def call(self, batch_size, ebno_db):
        return self.call_no_xla(batch_size, ebno_db)

    # Polar codes cannot be executed with XLA
    @tf.function(jit_compile=False) # enable graph mode 
    def call_no_xla(self, batch_size, ebno_db):
        
        u = self.source([batch_size, self.k]) # generate random data
        
        if self.encoder is None:
            # uncoded transmission
            c = u
        else:
            c = self.encoder(u) # explicitly encode

        # calculate noise variance
        if self.sim_esno:
            no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=1,
                       coderate=1)
        else: 
            if self.encoder is None:
                # uncoded transmission
                coderate = 1
            else:
                coderate = self.k/self.n

            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=self.num_bits_per_symbol,
                           coderate=coderate)            
        
        x = self.mapper(c) # map c to symbols x
        
        y = self.channel(x, no) # transmit over AWGN channel

        llr_ch = self.demapper(y, no) # demapp y to LLRs
        
        if self.decoder is None:
            # uncoded transmission
            u_hat = hard_decisions(llr_ch) 
        else:
            u_hat = self.decoder(llr_ch) # run FEC decoder (incl. rate-recovery)
        return u, u_hat
```

## Error Rate Simulations

We now compare the different schemes for a codeword length of $n=1024$ and coderate $r=0.5$.

Let us define the codes to be simulated.


```python
# code parameters
k = 512 # number of information bits per codeword
n = 1024 # desired codeword length
codes_under_test = []

# Uncoded transmission
enc = None
dec = None
name = "Uncoded QPSK"
codes_under_test.append([enc, dec, name])

# Conv. code with Viterbi decoding 
enc = ConvEncoder(rate=1/2, constraint_length=5)
dec = ViterbiDecoder(gen_poly=enc.gen_poly, method="soft_llr")
name = "GSM: Convolutional Codes"
codes_under_test.append([enc, dec, name])

# Turbo codes
enc = TurboEncoder(rate=1/2, constraint_length=4, terminate=True)
dec = TurboDecoder(encoder=enc, num_iter=8)
name = "UMTS/LTE: Turbo Codes"
codes_under_test.append([enc, dec, name])

# LDPC codes
enc = LDPC5GEncoder(k, n)
dec = LDPC5GDecoder(encoder=enc, num_iter=40)
name = "5G: LDPC"
codes_under_test.append([enc, dec, name])

# Polar codes
enc = Polar5GEncoder(k, n)
dec = Polar5GDecoder(enc, dec_type="hybSCL", list_size=32)
name = "5G: Polar+CRC"
codes_under_test.append([enc, dec, name])
```

Generate a new BER plot figure to save and plot simulation results efficiently.


```python
ber_plot = PlotBER("")
```

And run the BER simulation for each code.


```python
num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(0., 8, 0.2) # sim SNR range 

# run ber simulations for each code we have added to the list
for code in codes_under_test:
    print("\n Running: " + code[2])
    
    # generate a new model with the given encoder/decoder
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1],
                         sim_esno=False)
    
    # run the Polar code in a separate call, as currently no XLA is supported
    if not code[2]=="5G: Polar+CRC":
        ber_plot.simulate(model, # the function have defined previously
                        ebno_dbs=ebno_db, # SNR to simulate
                        legend=code[2], # legend string for plotting
                        max_mc_iter=1000, # run 1000 Monte Carlo runs per SNR point
                        num_target_block_errors=2000, # continue with next SNR point after 1000 bit errors
                        target_bler=3e-4,
                        batch_size=10000, # batch-size per Monte Carlo run
                        soft_estimates=False, # the model returns hard-estimates
                        early_stop=True, # stop simulation if no error has been detected at current SNR point
                        show_fig=False, # we show the figure after all results are simulated
                        add_bler=True, # in case BLER is also interesting
                        forward_keyboard_interrupt=False);
    else:
        # run model in non_xla mode        
        ber_plot.simulate(model.call_no_xla, # no XLA
                         ebno_dbs=ebno_db, # SNR to simulate
                         legend=code[2], # legend string for plotting
                         max_mc_iter=10000, # we use more iterations with smaller batches
                         num_target_block_errors=200, # continue with next SNR point after 1000 bit errors
                         target_bler=3e-4,
                         batch_size=1000, # batch-size per Monte Carlo run
                         soft_estimates=False, # the model returns hard-estimates
                         early_stop=True, # stop simulation if no error has been detected at current SNR point
                         show_fig=False, # we show the figure after all results are simulated
                         add_bler=True, # in case BLER is also interesting
                         forward_keyboard_interrupt=False);        
```

    
     Running: Uncoded QPSK


    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
          0.0 | 7.8639e-02 | 1.0000e+00 |      402630 |     5120000 |        10000 |       10000 |         1.9 |reached target block errors
          0.2 | 7.3970e-02 | 1.0000e+00 |      378725 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          0.4 | 6.9340e-02 | 1.0000e+00 |      355019 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          0.6 | 6.4879e-02 | 1.0000e+00 |      332180 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          0.8 | 6.0540e-02 | 1.0000e+00 |      309966 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          1.0 | 5.6242e-02 | 1.0000e+00 |      287957 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          1.2 | 5.2311e-02 | 1.0000e+00 |      267831 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          1.4 | 4.8157e-02 | 1.0000e+00 |      246564 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          1.6 | 4.4622e-02 | 1.0000e+00 |      228466 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          1.8 | 4.0873e-02 | 1.0000e+00 |      209271 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          2.0 | 3.7578e-02 | 1.0000e+00 |      192401 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          2.2 | 3.4334e-02 | 1.0000e+00 |      175790 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          2.4 | 3.1201e-02 | 1.0000e+00 |      159748 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          2.6 | 2.8221e-02 | 1.0000e+00 |      144492 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          2.8 | 2.5331e-02 | 1.0000e+00 |      129697 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          3.0 | 2.3013e-02 | 1.0000e+00 |      117825 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          3.2 | 2.0366e-02 | 9.9980e-01 |      104272 |     5120000 |         9998 |       10000 |         0.0 |reached target block errors
          3.4 | 1.8224e-02 | 1.0000e+00 |       93309 |     5120000 |        10000 |       10000 |         0.0 |reached target block errors
          3.6 | 1.6016e-02 | 9.9980e-01 |       82001 |     5120000 |         9998 |       10000 |         0.0 |reached target block errors
          3.8 | 1.4282e-02 | 9.9930e-01 |       73124 |     5120000 |         9993 |       10000 |         0.0 |reached target block errors
          4.0 | 1.2508e-02 | 9.9850e-01 |       64042 |     5120000 |         9985 |       10000 |         0.0 |reached target block errors
          4.2 | 1.0846e-02 | 9.9640e-01 |       55532 |     5120000 |         9964 |       10000 |         0.0 |reached target block errors
          4.4 | 9.4846e-03 | 9.9220e-01 |       48561 |     5120000 |         9922 |       10000 |         0.0 |reached target block errors
          4.6 | 8.0947e-03 | 9.8530e-01 |       41445 |     5120000 |         9853 |       10000 |         0.0 |reached target block errors
          4.8 | 7.0646e-03 | 9.7390e-01 |       36171 |     5120000 |         9739 |       10000 |         0.0 |reached target block errors
          5.0 | 5.9846e-03 | 9.5410e-01 |       30641 |     5120000 |         9541 |       10000 |         0.0 |reached target block errors
          5.2 | 5.0121e-03 | 9.2440e-01 |       25662 |     5120000 |         9244 |       10000 |         0.0 |reached target block errors
          5.4 | 4.2078e-03 | 8.9130e-01 |       21544 |     5120000 |         8913 |       10000 |         0.0 |reached target block errors
          5.6 | 3.4760e-03 | 8.2980e-01 |       17797 |     5120000 |         8298 |       10000 |         0.0 |reached target block errors
          5.8 | 2.8916e-03 | 7.7470e-01 |       14805 |     5120000 |         7747 |       10000 |         0.0 |reached target block errors
          6.0 | 2.3920e-03 | 7.0570e-01 |       12247 |     5120000 |         7057 |       10000 |         0.0 |reached target block errors
          6.2 | 1.9475e-03 | 6.3220e-01 |        9971 |     5120000 |         6322 |       10000 |         0.0 |reached target block errors
          6.4 | 1.5863e-03 | 5.5570e-01 |        8122 |     5120000 |         5557 |       10000 |         0.0 |reached target block errors
          6.6 | 1.2424e-03 | 4.7520e-01 |        6361 |     5120000 |         4752 |       10000 |         0.0 |reached target block errors
          6.8 | 9.8926e-04 | 3.9790e-01 |        5065 |     5120000 |         3979 |       10000 |         0.0 |reached target block errors
          7.0 | 7.6328e-04 | 3.2270e-01 |        3908 |     5120000 |         3227 |       10000 |         0.0 |reached target block errors
          7.2 | 5.9531e-04 | 2.6300e-01 |        3048 |     5120000 |         2630 |       10000 |         0.0 |reached target block errors
          7.4 | 4.5977e-04 | 2.0680e-01 |        2354 |     5120000 |         2068 |       10000 |         0.0 |reached target block errors
          7.6 | 3.3760e-04 | 1.5880e-01 |        3457 |    10240000 |         3176 |       20000 |         0.0 |reached target block errors
          7.8 | 2.5752e-04 | 1.2340e-01 |        2637 |    10240000 |         2468 |       20000 |         0.0 |reached target block errors
    
     Running: GSM: Convolutional Codes


    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
          0.0 | 1.1188e-01 | 1.0000e+00 |      572839 |     5120000 |        10000 |       10000 |         7.3 |reached target block errors
          0.2 | 9.5131e-02 | 9.9970e-01 |      487069 |     5120000 |         9997 |       10000 |         0.1 |reached target block errors
          0.4 | 7.9436e-02 | 9.9900e-01 |      406711 |     5120000 |         9990 |       10000 |         0.1 |reached target block errors
          0.6 | 6.4824e-02 | 9.9620e-01 |      331900 |     5120000 |         9962 |       10000 |         0.1 |reached target block errors
          0.8 | 5.2889e-02 | 9.9280e-01 |      270794 |     5120000 |         9928 |       10000 |         0.1 |reached target block errors
          1.0 | 4.1660e-02 | 9.7820e-01 |      213301 |     5120000 |         9782 |       10000 |         0.1 |reached target block errors
          1.2 | 3.2918e-02 | 9.5830e-01 |      168542 |     5120000 |         9583 |       10000 |         0.1 |reached target block errors
          1.4 | 2.5616e-02 | 9.2220e-01 |      131153 |     5120000 |         9222 |       10000 |         0.1 |reached target block errors
          1.6 | 1.9042e-02 | 8.5830e-01 |       97497 |     5120000 |         8583 |       10000 |         0.1 |reached target block errors
          1.8 | 1.4026e-02 | 7.9010e-01 |       71815 |     5120000 |         7901 |       10000 |         0.1 |reached target block errors
          2.0 | 1.0373e-02 | 6.9010e-01 |       53111 |     5120000 |         6901 |       10000 |         0.1 |reached target block errors
          2.2 | 7.4807e-03 | 5.9100e-01 |       38301 |     5120000 |         5910 |       10000 |         0.1 |reached target block errors
          2.4 | 5.4088e-03 | 4.9580e-01 |       27693 |     5120000 |         4958 |       10000 |         0.1 |reached target block errors
          2.6 | 3.6822e-03 | 3.9570e-01 |       18853 |     5120000 |         3957 |       10000 |         0.1 |reached target block errors
          2.8 | 2.6064e-03 | 3.1930e-01 |       13345 |     5120000 |         3193 |       10000 |         0.1 |reached target block errors
          3.0 | 1.7748e-03 | 2.3750e-01 |        9087 |     5120000 |         2375 |       10000 |         0.1 |reached target block errors
          3.2 | 1.2456e-03 | 1.8075e-01 |       12755 |    10240000 |         3615 |       20000 |         0.1 |reached target block errors
          3.4 | 8.4326e-04 | 1.3490e-01 |        8635 |    10240000 |         2698 |       20000 |         0.1 |reached target block errors
          3.6 | 5.7129e-04 | 1.0220e-01 |        5850 |    10240000 |         2044 |       20000 |         0.1 |reached target block errors
          3.8 | 3.7826e-04 | 7.4767e-02 |        5810 |    15360000 |         2243 |       30000 |         0.2 |reached target block errors
          4.0 | 2.6528e-04 | 5.5600e-02 |        5433 |    20480000 |         2224 |       40000 |         0.2 |reached target block errors
          4.2 | 1.8230e-04 | 4.1580e-02 |        4667 |    25600000 |         2079 |       50000 |         0.3 |reached target block errors
          4.4 | 1.2107e-04 | 3.0243e-02 |        4339 |    35840000 |         2117 |       70000 |         0.4 |reached target block errors
          4.6 | 8.1185e-05 | 2.2300e-02 |        3741 |    46080000 |         2007 |       90000 |         0.5 |reached target block errors
          4.8 | 6.0286e-05 | 1.7058e-02 |        3704 |    61440000 |         2047 |      120000 |         0.7 |reached target block errors
          5.0 | 4.4805e-05 | 1.3413e-02 |        3441 |    76800000 |         2012 |      150000 |         0.9 |reached target block errors
          5.2 | 3.2061e-05 | 1.0450e-02 |        3283 |   102400000 |         2090 |      200000 |         1.2 |reached target block errors
          5.4 | 2.4186e-05 | 8.4750e-03 |        2972 |   122880000 |         2034 |      240000 |         1.4 |reached target block errors
          5.6 | 1.6527e-05 | 6.0676e-03 |        2877 |   174080000 |         2063 |      340000 |         2.0 |reached target block errors
          5.8 | 1.2327e-05 | 4.7047e-03 |        2714 |   220160000 |         2023 |      430000 |         2.5 |reached target block errors
          6.0 | 9.9204e-06 | 3.8264e-03 |        2692 |   271360000 |         2028 |      530000 |         3.0 |reached target block errors
          6.2 | 7.2783e-06 | 2.9441e-03 |        2534 |   348160000 |         2002 |      680000 |         3.9 |reached target block errors
          6.4 | 5.5864e-06 | 2.2818e-03 |        2517 |   450560000 |         2008 |      880000 |         5.1 |reached target block errors
          6.6 | 4.1099e-06 | 1.7231e-03 |        2462 |   599040000 |         2016 |     1170000 |         6.7 |reached target block errors
          6.8 | 2.9948e-06 | 1.3105e-03 |        2346 |   783360000 |         2005 |     1530000 |         8.8 |reached target block errors
          7.0 | 2.2300e-06 | 9.7718e-04 |        2352 |  1054720000 |         2013 |     2060000 |        11.8 |reached target block errors
          7.2 | 1.6952e-06 | 7.6374e-04 |        2274 |  1341440000 |         2001 |     2620000 |        15.1 |reached target block errors
          7.4 | 1.2223e-06 | 5.5694e-04 |        2253 |  1843200000 |         2005 |     3600000 |        20.7 |reached target block errors
          7.6 | 9.2857e-07 | 4.2756e-04 |        2225 |  2396160000 |         2001 |     4680000 |        27.0 |reached target block errors
          7.8 | 6.5748e-07 | 3.0989e-04 |        2178 |  3312640000 |         2005 |     6470000 |        37.4 |reached target block errors
    
     Running: UMTS/LTE: Turbo Codes
    <dtype: 'float32'>


    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
          0.0 | 1.2036e-01 | 9.8210e-01 |      616241 |     5120000 |         9821 |       10000 |        47.3 |reached target block errors
          0.2 | 1.0284e-01 | 9.3380e-01 |      526521 |     5120000 |         9338 |       10000 |         0.5 |reached target block errors
          0.4 | 8.0533e-02 | 8.3220e-01 |      412327 |     5120000 |         8322 |       10000 |         0.5 |reached target block errors
          0.6 | 5.4910e-02 | 6.3940e-01 |      281140 |     5120000 |         6394 |       10000 |         0.5 |reached target block errors
          0.8 | 3.2185e-02 | 4.1190e-01 |      164786 |     5120000 |         4119 |       10000 |         0.5 |reached target block errors
          1.0 | 1.6551e-02 | 2.2840e-01 |       84740 |     5120000 |         2284 |       10000 |         0.5 |reached target block errors
          1.2 | 6.2887e-03 | 9.6700e-02 |       96595 |    15360000 |         2901 |       30000 |         1.5 |reached target block errors
          1.4 | 1.9281e-03 | 3.2257e-02 |       69104 |    35840000 |         2258 |       70000 |         3.5 |reached target block errors
          1.6 | 4.9819e-04 | 9.1227e-03 |       56116 |   112640000 |         2007 |      220000 |        11.1 |reached target block errors
          1.8 | 9.6408e-05 | 2.0629e-03 |       47880 |   496640000 |         2001 |      970000 |        48.9 |reached target block errors
          2.0 | 1.5120e-05 | 3.9683e-04 |       39016 |  2580480000 |         2000 |     5040000 |       254.2 |reached target block errors
          2.2 | 2.2768e-06 | 8.2700e-05 |       11657 |  5120000000 |          827 |    10000000 |       504.0 |reached max iterations
    
    Simulation stopped as target BLER is reached @ EbNo = 2.2 dB.
    
    
     Running: 5G: LDPC


    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
          0.0 | 1.6374e-01 | 9.8460e-01 |      838357 |     5120000 |         9846 |       10000 |         6.8 |reached target block errors
          0.2 | 1.4213e-01 | 9.4050e-01 |      727682 |     5120000 |         9405 |       10000 |         0.2 |reached target block errors
          0.4 | 1.1441e-01 | 8.3600e-01 |      585782 |     5120000 |         8360 |       10000 |         0.2 |reached target block errors
          0.6 | 8.2134e-02 | 6.5360e-01 |      420528 |     5120000 |         6536 |       10000 |         0.2 |reached target block errors
          0.8 | 5.0532e-02 | 4.3360e-01 |      258724 |     5120000 |         4336 |       10000 |         0.2 |reached target block errors
          1.0 | 2.6273e-02 | 2.4060e-01 |      134516 |     5120000 |         2406 |       10000 |         0.2 |reached target block errors
          1.2 | 1.0823e-02 | 1.0285e-01 |      110829 |    10240000 |         2057 |       20000 |         0.4 |reached target block errors
          1.4 | 3.5725e-03 | 3.6383e-02 |      109748 |    30720000 |         2183 |       60000 |         1.1 |reached target block errors
          1.6 | 1.0070e-03 | 1.0647e-02 |       97958 |    97280000 |         2023 |      190000 |         3.5 |reached target block errors
          1.8 | 2.1172e-04 | 2.3729e-03 |       92141 |   435200000 |         2017 |      850000 |        15.8 |reached target block errors
          2.0 | 3.6021e-05 | 4.5545e-04 |       81149 |  2252800000 |         2004 |     4400000 |        82.1 |reached target block errors
          2.2 | 5.1234e-06 | 8.8700e-05 |       26232 |  5120000000 |          887 |    10000000 |       186.6 |reached max iterations
    
    Simulation stopped as target BLER is reached @ EbNo = 2.2 dB.
    
    
     Running: 5G: Polar+CRC
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
          0.0 | 4.1038e-01 | 9.3900e-01 |      210117 |      512000 |          939 |        1000 |       724.1 |reached target block errors
          0.2 | 3.4673e-01 | 8.3400e-01 |      177524 |      512000 |          834 |        1000 |       666.8 |reached target block errors
          0.4 | 2.7139e-01 | 6.8300e-01 |      138950 |      512000 |          683 |        1000 |       666.2 |reached target block errors
          0.6 | 1.7986e-01 | 4.7800e-01 |       92090 |      512000 |          478 |        1000 |       640.0 |reached target block errors
          0.8 | 9.5201e-02 | 2.9000e-01 |       48743 |      512000 |          290 |        1000 |       604.9 |reached target block errors
          1.0 | 4.7203e-02 | 1.4900e-01 |       48336 |     1024000 |          298 |        2000 |      1096.8 |reached target block errors
          1.2 | 1.4817e-02 | 5.0500e-02 |       30346 |     2048000 |          202 |        4000 |      1801.8 |reached target block errors
          1.4 | 5.0508e-03 | 1.7250e-02 |       31032 |     6144000 |          207 |       12000 |      4091.6 |reached target block errors
          1.6 | 1.0900e-03 | 4.4130e-03 |       25672 |    23552000 |          203 |       46000 |     10743.4 |reached target block errors
          1.8 | 2.9683e-04 | 1.1802e-03 |       26140 |    88064000 |          203 |      172000 |     24898.3 |reached target block errors
          2.0 | 5.4104e-05 | 2.5221e-04 |       21967 |   406016000 |          200 |      793000 |     66052.9 |reached target block errors
    
    Simulation stopped as target BLER is reached @ EbNo = 2.0 dB.
    


And show the final performance


```python
# remove "(BLER)" labels from legend
for idx, l in enumerate(ber_plot.legend):
    ber_plot.legend[idx] = l.replace(" (BLER)", "")
    
# and plot the BLER
ber_plot(xlim=[0, 7], ylim=[3.e-4, 1], show_ber=False)
```


    
![png](Evolution_of_FEC_files/Evolution_of_FEC_13_0.png)
    



```python
# BER
ber_plot(xlim=[0, 7], ylim=[2.e-7, 1], show_bler=False)
```


    
![png](Evolution_of_FEC_files/Evolution_of_FEC_14_0.png)
    


## Results for Longer Codewords

In particular for the data channels, longer codewords are usually required.
For these applications, LDPC and Turbo codes are the workhorse of 5G and LTE, respectively. 

Let's compare LDPC and Turbo codes for $k=6144$ information bits and coderate $r=1/3$.


```python
# code parameters
k = 2048 # number of information bits per codeword
n = 6156 # desired codeword length (including termination bits)
codes_under_test = []

# Uncoded QPSK
enc = None
dec = None
name = "Uncoded QPSK"
codes_under_test.append([enc, dec, name])

#Turbo. codes
enc = TurboEncoder(rate=1/3, constraint_length=4, terminate=True)
dec = TurboDecoder(encoder=enc, num_iter=8)
name = "UMTS/LTE: Turbo Codes"
codes_under_test.append([enc, dec, name])

# LDPC
enc = LDPC5GEncoder(k, n)
dec = LDPC5GDecoder(encoder=enc, num_iter=40)
name = "5G: LDPC"
codes_under_test.append([enc, dec, name])
```


```python
ber_plot_long = PlotBER(f"Error Rate Performance (k={k}, n={n})")
```


```python
num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(-1, 1.8, 0.1) # sim SNR range 

# run ber simulations for each code we have added to the list
for code in codes_under_test:
    print("\n Running: " + code[2])
    
    # generate a new model with the given encoder/decoder
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1],
                         sim_esno=False)
    
    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    ber_plot_long.simulate(model, # the function have defined previously
                     ebno_dbs=ebno_db, # SNR to simulate
                     legend=code[2], # legend string for plotting
                     max_mc_iter=1000, # run 100 Monte Carlo runs per SNR point
                     num_target_block_errors=500, # continue with next SNR point after 2000 bit errors
                     target_ber=6e-7,
                     batch_size=10000, # batch-size per Monte Carlo run
                     soft_estimates=False, # the model returns hard-estimates
                     early_stop=True, # stop simulation if no error has been detected at current SNR point
                     show_fig=False, # we show the figure after all results are simulated
                     add_bler=True, # in case BLER is also interesting
                     forward_keyboard_interrupt=False); # should be True in a loop
```

    
     Running: Uncoded QPSK
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
         -1.0 | 1.0363e-01 | 1.0000e+00 |     2122436 |    20480000 |        10000 |       10000 |         1.2 |reached target block errors
         -0.9 | 1.0124e-01 | 1.0000e+00 |     2073392 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
         -0.8 | 9.8453e-02 | 1.0000e+00 |     2016326 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
         -0.7 | 9.5972e-02 | 1.0000e+00 |     1965499 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
         -0.6 | 9.3551e-02 | 1.0000e+00 |     1915924 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
         -0.5 | 9.1005e-02 | 1.0000e+00 |     1863781 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
         -0.4 | 8.8387e-02 | 1.0000e+00 |     1810164 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
         -0.3 | 8.5901e-02 | 1.0000e+00 |     1759256 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
         -0.2 | 8.3391e-02 | 1.0000e+00 |     1707855 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
         -0.1 | 8.1089e-02 | 1.0000e+00 |     1660697 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
         -0.0 | 7.8586e-02 | 1.0000e+00 |     1609442 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          0.1 | 7.6298e-02 | 1.0000e+00 |     1562589 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          0.2 | 7.3904e-02 | 1.0000e+00 |     1513552 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          0.3 | 7.1568e-02 | 1.0000e+00 |     1465710 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          0.4 | 6.9284e-02 | 1.0000e+00 |     1418945 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          0.5 | 6.7130e-02 | 1.0000e+00 |     1374818 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          0.6 | 6.4722e-02 | 1.0000e+00 |     1325503 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          0.7 | 6.2659e-02 | 1.0000e+00 |     1283261 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          0.8 | 6.0441e-02 | 1.0000e+00 |     1237828 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          0.9 | 5.8366e-02 | 1.0000e+00 |     1195334 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          1.0 | 5.6213e-02 | 1.0000e+00 |     1151249 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          1.1 | 5.4286e-02 | 1.0000e+00 |     1111775 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          1.2 | 5.2204e-02 | 1.0000e+00 |     1069128 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          1.3 | 5.0238e-02 | 1.0000e+00 |     1028882 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          1.4 | 4.8339e-02 | 1.0000e+00 |      989989 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          1.5 | 4.6420e-02 | 1.0000e+00 |      950689 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          1.6 | 4.4479e-02 | 1.0000e+00 |      910934 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
          1.7 | 4.2722e-02 | 1.0000e+00 |      874952 |    20480000 |        10000 |       10000 |         0.0 |reached target block errors
    
     Running: UMTS/LTE: Turbo Codes
    <dtype: 'float32'>
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
         -1.0 | 1.9316e-01 | 1.0000e+00 |     3955956 |    20480000 |        10000 |       10000 |        91.5 |reached target block errors
         -0.9 | 1.8637e-01 | 1.0000e+00 |     3816948 |    20480000 |        10000 |       10000 |         2.0 |reached target block errors
         -0.8 | 1.7884e-01 | 1.0000e+00 |     3662551 |    20480000 |        10000 |       10000 |         2.0 |reached target block errors
         -0.7 | 1.7090e-01 | 1.0000e+00 |     3500009 |    20480000 |        10000 |       10000 |         2.0 |reached target block errors
         -0.6 | 1.6183e-01 | 1.0000e+00 |     3314222 |    20480000 |        10000 |       10000 |         2.0 |reached target block errors
         -0.5 | 1.5148e-01 | 1.0000e+00 |     3102411 |    20480000 |        10000 |       10000 |         2.0 |reached target block errors
         -0.4 | 1.3943e-01 | 9.9930e-01 |     2855562 |    20480000 |         9993 |       10000 |         2.0 |reached target block errors
         -0.3 | 1.2454e-01 | 9.9460e-01 |     2550626 |    20480000 |         9946 |       10000 |         2.0 |reached target block errors
         -0.2 | 1.0651e-01 | 9.7720e-01 |     2181304 |    20480000 |         9772 |       10000 |         2.0 |reached target block errors
         -0.1 | 8.4806e-02 | 9.1860e-01 |     1736827 |    20480000 |         9186 |       10000 |         2.0 |reached target block errors
         -0.0 | 5.9478e-02 | 7.9110e-01 |     1218100 |    20480000 |         7911 |       10000 |         2.0 |reached target block errors
          0.1 | 3.6126e-02 | 5.9480e-01 |      739869 |    20480000 |         5948 |       10000 |         2.0 |reached target block errors
          0.2 | 1.7707e-02 | 3.5840e-01 |      362633 |    20480000 |         3584 |       10000 |         2.0 |reached target block errors
          0.3 | 7.0990e-03 | 1.7600e-01 |      145387 |    20480000 |         1760 |       10000 |         2.0 |reached target block errors
          0.4 | 2.0591e-03 | 6.2600e-02 |       42170 |    20480000 |          626 |       10000 |         2.0 |reached target block errors
          0.5 | 5.1164e-04 | 2.0200e-02 |       31435 |    61440000 |          606 |       30000 |         6.1 |reached target block errors
          0.6 | 1.0209e-04 | 4.6273e-03 |       22999 |   225280000 |          509 |      110000 |        22.4 |reached target block errors
          0.7 | 1.5876e-05 | 9.0536e-04 |       18208 |  1146880000 |          507 |      560000 |       114.6 |reached target block errors
          0.8 | 2.0979e-06 | 1.4493e-04 |       14823 |  7065600000 |          500 |     3450000 |       707.5 |reached target block errors
          0.9 | 2.3550e-07 | 2.9200e-05 |        4823 | 20480000000 |          292 |    10000000 |      2049.4 |reached max iterations
    
    Simulation stopped as target BER is reached@ EbNo = 0.9 dB.
    
    
     Running: 5G: LDPC
    EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
    ---------------------------------------------------------------------------------------------------------------------------------------
         -1.0 | 2.3049e-01 | 1.0000e+00 |     4720446 |    20480000 |        10000 |       10000 |         9.3 |reached target block errors
         -0.9 | 2.2409e-01 | 1.0000e+00 |     4589269 |    20480000 |        10000 |       10000 |         1.0 |reached target block errors
         -0.8 | 2.1685e-01 | 1.0000e+00 |     4441089 |    20480000 |        10000 |       10000 |         1.0 |reached target block errors
         -0.7 | 2.0853e-01 | 1.0000e+00 |     4270774 |    20480000 |        10000 |       10000 |         1.0 |reached target block errors
         -0.6 | 1.9817e-01 | 9.9980e-01 |     4058446 |    20480000 |         9998 |       10000 |         1.0 |reached target block errors
         -0.5 | 1.8486e-01 | 9.9900e-01 |     3785895 |    20480000 |         9990 |       10000 |         1.0 |reached target block errors
         -0.4 | 1.6716e-01 | 9.9250e-01 |     3423397 |    20480000 |         9925 |       10000 |         1.0 |reached target block errors
         -0.3 | 1.4083e-01 | 9.6620e-01 |     2884180 |    20480000 |         9662 |       10000 |         1.0 |reached target block errors
         -0.2 | 1.1100e-01 | 8.9430e-01 |     2273316 |    20480000 |         8943 |       10000 |         1.0 |reached target block errors
         -0.1 | 7.5297e-02 | 7.4070e-01 |     1542084 |    20480000 |         7407 |       10000 |         1.0 |reached target block errors
         -0.0 | 4.4597e-02 | 5.2670e-01 |      913339 |    20480000 |         5267 |       10000 |         1.0 |reached target block errors
          0.1 | 2.1984e-02 | 3.1200e-01 |      450224 |    20480000 |         3120 |       10000 |         1.0 |reached target block errors
          0.2 | 8.2926e-03 | 1.3900e-01 |      169833 |    20480000 |         1390 |       10000 |         1.0 |reached target block errors
          0.3 | 2.5131e-03 | 5.0500e-02 |       51469 |    20480000 |          505 |       10000 |         1.0 |reached target block errors
          0.4 | 6.7440e-04 | 1.4300e-02 |       55247 |    81920000 |          572 |       40000 |         4.1 |reached target block errors
          0.5 | 1.3451e-04 | 3.3333e-03 |       41321 |   307200000 |          500 |      150000 |        15.2 |reached target block errors
          0.6 | 2.5533e-05 | 6.5789e-04 |       39742 |  1556480000 |          500 |      760000 |        77.1 |reached target block errors
          0.7 | 2.8272e-06 | 1.2255e-04 |       23624 |  8355840000 |          500 |     4080000 |       414.4 |reached target block errors
          0.8 | 5.6611e-07 | 4.0100e-05 |       11594 | 20480000000 |          401 |    10000000 |      1015.1 |reached max iterations
    
    Simulation stopped as target BER is reached@ EbNo = 0.8 dB.
    



```python
# and show the figure
ber_plot_long(xlim=[-1., 1.7],ylim=(6e-7, 1)) # we set the ylim to 1e-5 as otherwise more extensive simualtions would be required for accurate curves.
```


    
![png](Evolution_of_FEC_files/Evolution_of_FEC_19_0.png)
    


A comparison of short length codes can be found in the tutorial notebook [5G Channel Coding Polar vs. LDPC Codes](5G_Channel_Coding_Polar_vs_LDPC_Codes.ipynb).

## Final Figure

Combine results from the two simulations.


```python
snrs = list(np.compress(a=ber_plot._snrs, condition=ber_plot._is_bler, axis=0))
bers = list(np.compress(a=ber_plot._bers, condition=ber_plot._is_bler, axis=0))
legends = list(np.compress(a=ber_plot._legends, condition=ber_plot._is_bler, axis=0))
is_bler = list(np.compress(a=ber_plot._is_bler, condition=ber_plot._is_bler, axis=0))

ylabel = "BLER"

# generate two subplots
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,10))

ax1.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)

   
# Part A 
xlim=[0, 6]
ylim=[1e-4, 1]

ax1.set_xlim(xlim)
ax1.set_ylim(ylim)

for idx, b in enumerate(bers):
    ax1.semilogy(snrs[idx], b, "--", linewidth=2)

ax1.grid(which="both")
ax1.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
ax1.set_ylabel(ylabel, fontsize=25)
ax1.legend(legends, fontsize=20, loc="upper right");
ax1.set_title("$k=512, n=1024$", fontsize=20)


# remove "(BLER)" labels from legend
for idx, l in enumerate(ber_plot_long.legend):
    ber_plot_long.legend[idx] = l.replace(" (BLER)", "")
    
snrs = list(np.compress(condition=ber_plot_long._is_bler, a=ber_plot_long._snrs, axis=0))
bers = list(np.compress(condition=ber_plot_long._is_bler, a=ber_plot_long._bers, axis=0))
legends = list(np.compress(condition=ber_plot_long._is_bler, a=ber_plot_long._legends, axis=0))
is_bler = list(np.compress(condition=ber_plot_long._is_bler, a=ber_plot_long._is_bler, axis=0))


# Part B
xlim=[-1, 2]
ylim=[1e-4, 1]

ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_title("$k=2048, n=6156$", fontsize=20)

# return figure handle
#for idx, b in enumerate(bers):

ax2.semilogy(snrs[0], bers[0], "--", linewidth=2, color="orange")
ax2.semilogy(snrs[1], bers[1], "--", linewidth=2, color="green")
ax2.semilogy(snrs[2], bers[2], "--", linewidth=2, color="blue")

ax2.grid(which="both")
ax2.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
ax2.set_ylabel(ylabel, fontsize=25)
plt.legend(legends, fontsize=20, loc="upper right");
```


    
![png](Evolution_of_FEC_files/Evolution_of_FEC_22_0.png)
    

