# Multiple-Input Multiple-Output (MIMO)

## Stream Management

Stream management determines which transmitter is sending which stream
to which receiver. Transmitters and receivers can be user terminals or
base stations, depending on whether uplink or downlink transmissions are
considered. The `~sionna.phy.mimo.StreamManagement` class has various
properties that are needed to recover desired or interfering channel
coefficients for precoding and equalization. In order to understand how
the various properties of `~sionna.phy.mimo.StreamManagement` can be
used, we recommend to have a look at the source code of the
`~sionna.phy.ofdm.LMMSEEqualizer` or `~sionna.phy.ofdm.RZFPrecoder`.

The following code snippet shows how to configure
`~sionna.phy.mimo.StreamManagement` for a simple uplink scenario, where
four transmitters send each one stream to a receiver. Note that
`~sionna.phy.mimo.StreamManagement` is independent of the actual number
of antennas at the transmitters and receivers.

``` Python
num_tx = 4
num_rx = 1
num_streams_per_tx = 1

# Indicate which transmitter is associated with which receiver
# rx_tx_association[i,j] = 1 means that transmitter j sends one
# or mutiple streams to receiver i.
rx_tx_association = np.zeros([num_rx, num_tx])
rx_tx_association[0,0] = 1
rx_tx_association[0,1] = 1
rx_tx_association[0,2] = 1
rx_tx_association[0,3] = 1

sm = StreamManagement(rx_tx_association, num_streams_per_tx)
```

<div class="autoclass"
members="precoding_ind, rx_tx_association, num_rx, num_tx, num_streams_per_tx, num_streams_per_rx, num_interfering_streams_per_rx, num_tx_per_rx, num_rx_per_tx, stream_association, detection_desired_ind, detection_undesired_ind, tx_stream_ids, rx_stream_ids, stream_ind">

sionna.phy.mimo.StreamManagement

</div>

## Precoding

<div class="autofunction">

sionna.phy.mimo.cbf_precoding_matrix

</div>

<div class="autofunction">

sionna.phy.mimo.rzf_precoding_matrix

</div>

<div class="autofunction">

sionna.phy.mimo.rzf_precoder

</div>

<div class="autofunction">

sionna.phy.mimo.grid_of_beams_dft_ula

</div>

<div class="autofunction">

sionna.phy.mimo.grid_of_beams_dft

</div>

<div class="autofunction">

sionna.phy.mimo.flatten_precoding_mat

</div>

<div class="autofunction">

sionna.phy.mimo.normalize_precoding_power

</div>

## Equalization

<div class="autofunction">

sionna.phy.mimo.lmmse_matrix

</div>

<div class="autofunction">

sionna.phy.mimo.lmmse_equalizer

</div>

<div class="autofunction">

sionna.phy.mimo.mf_equalizer

</div>

<div class="autofunction">

sionna.phy.mimo.zf_equalizer

</div>

## Detection

<div class="autoclass"
exclude-members="call, build, compute_sigma_mu, compute_v_x, compute_v_x_obs, update_lam_gam"
members="">

sionna.phy.mimo.EPDetector

</div>

<div class="autoclass" exclude-members="call, build" members="">

sionna.phy.mimo.KBestDetector

</div>

<div class="autoclass" exclude-members="call, build" members="">

sionna.phy.mimo.LinearDetector

</div>

<div class="autoclass" exclude-members="call, build" members="">

sionna.phy.mimo.MaximumLikelihoodDetector

</div>

<div class="autoclass" exclude-members="call, build" members="">

sionna.phy.mimo.MMSEPICDetector

</div>

## Utility Functions

<div class="autoclass" exclude-members="__call__" members="">

sionna.phy.mimo.List2LLR

</div>

<div class="autoclass" exclude-members="call, build" members="">

sionna.phy.mimo.List2LLRSimple

</div>

<div class="autofunction">

sionna.phy.mimo.complex2real_vector

</div>

<div class="autofunction">

sionna.phy.mimo.real2complex_vector

</div>

<div class="autofunction">

sionna.phy.mimo.complex2real_matrix

</div>

<div class="autofunction">

sionna.phy.mimo.real2complex_matrix

</div>

<div class="autofunction">

sionna.phy.mimo.complex2real_covariance

</div>

<div class="autofunction">

sionna.phy.mimo.real2complex_covariance

</div>

<div class="autofunction">

sionna.phy.mimo.complex2real_channel

</div>

<div class="autofunction">

sionna.phy.mimo.real2complex_channel

</div>

<div class="autofunction">

sionna.phy.mimo.whiten_channel

</div>

References:  

<div id="citations">

<span id="CST2011" class="citation-label">CST2011</span>  
C. Studer, S. Fateh, and D. Seethaler, ["ASIC Implementation of
Soft-Input Soft-Output MIMO Detection Using MMSE Parallel Interference
Cancellation"](https://ieeexplore.ieee.org/abstract/document/5779722),
IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765,
July 2011.

<span id="CovProperRV" class="citation-label">CovProperRV</span>  
[Covariance matrices of real and imaginary
parts](https://en.wikipedia.org/wiki/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts),
Wikipedia, accessed 11 September, 2022.

<span id="EP2014" class="citation-label">EP2014</span>  
J. Céspedes, P. M. Olmos, M. Sánchez-Fernández, and F. Perez-Cruz,
["Expectation Propagation Detection for High-Order High-Dimensional MIMO
Systems"](https://ieeexplore.ieee.org/abstract/document/6841617), IEEE
Trans. Commun., vol. 62, no. 8, pp. 2840-2849, Aug. 2014.

<span id="FT2015" class="citation-label">FT2015</span>  
W. Fu and J. S. Thompson, ["Performance analysis of K-best detection
with adaptive
modulation"](https://ieeexplore.ieee.org/abstract/document/7454351),
IEEE Int. Symp. Wirel. Commun. Sys. (ISWCS), 2015.

<span id="ProperRV" class="citation-label">ProperRV</span>  
[Proper complex random
variables](https://en.wikipedia.org/wiki/Complex_random_variable#Proper_complex_random_variables),
Wikipedia, accessed 11 September, 2022.

<span id="YH2015" class="citation-label">YH2015</span>  
S. Yang and L. Hanzo, ["Fifty Years of MIMO Detection: The Road to
Large-Scale
MIMOs"](https://ieeexplore.ieee.org/abstract/document/7244171), IEEE
Communications Surveys & Tutorials, vol. 17, no. 4, pp. 1941-1988, 2015.

</div>
