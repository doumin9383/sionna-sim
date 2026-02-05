# Wireless Channel Models

This module provides blocks and functions that implement wireless
channel models. Models currently available include
`~sionna.phy.channel.AWGN`, `flat-fading <flat-fading>` with (optional)
`~sionna.phy.channel.SpatialCorrelation`,
`~sionna.phy.channel.RayleighBlockFading`, as well as models from the
3rd Generation Partnership Project (3GPP)
<a href="#TR38901" class="citation">[TR38901]</a>: `TDL <tdl>`,
`CDL <cdl>`, `UMi <umi>`, `UMa <uma>`, and `RMa <rma>`. It is also
possible to `use externally generated CIRs <external-datasets>`.

Apart from `flat-fading <flat-fading>`, all of these models generate
channel impulse responses (CIRs) that can then be used to implement a
channel transfer function in the `time domain <time-domain>` or
`assuming an OFDM waveform <ofdm-waveform>`.

This is achieved using the different functions, classes, and Keras
layers which operate as shown in the figures below.

<figure class="align-center">
<img src="../figures/channel_arch_time.png"
alt="Channel module architecture for time domain simulations." />
<figcaption aria-hidden="true">Channel module architecture for time
domain simulations.</figcaption>
</figure>

<figure class="align-center">
<img src="../figures/channel_arch_freq.png"
alt="Channel module architecture for simulations assuming OFDM waveform." />
<figcaption aria-hidden="true">Channel module architecture for
simulations assuming OFDM waveform.</figcaption>
</figure>

A channel model generate CIRs from which channel responses in the time
domain or in the frequency domain are computed using the
`~sionna.phy.channel.cir_to_time_channel` or
`~sionna.phy.channel.cir_to_ofdm_channel` functions, respectively. If
one does not need access to the raw CIRs, the
`~sionna.phy.channel.GenerateTimeChannel` and
`~sionna.phy.channel.GenerateOFDMChannel` classes can be used to
conveniently sample CIRs and generate channel responses in the desired
domain.

Once the channel responses in the time or frequency domain are computed,
they can be applied to the channel input using the
`~sionna.phy.channel.ApplyTimeChannel` or
`~sionna.phy.channel.ApplyOFDMChannel` Keras layers.

The following code snippets show how to setup and run a Rayleigh block
fading model assuming an OFDM waveform, and without accessing the CIRs
or channel responses. This is the easiest way to setup a channel model.
Setting-up other models is done in a similar way, except for
`~sionna.phy.channel.AWGN` (see the `~sionna.phy.channel.AWGN` class
documentation).

``` Python
rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 32,
                               num_tx = 4,
                               num_tx_ant = 2)

channel  = OFDMChannel(channel_model = rayleigh,
                       resource_grid = rg)
```

where `rg` is an instance of `~sionna.phy.ofdm.ResourceGrid`.

Running the channel model is done as follows:

``` Python
# x is the channel input
# no is the noise variance
y = channel([x, no])
```

To use the time domain representation of the channel, one can use
`~sionna.phy.channel.TimeChannel` instead of
`~sionna.phy.channel.OFDMChannel`.

If access to the channel responses is needed, one can separate their
generation from their application to the channel input by setting up the
channel model as follows:

``` Python
rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 32,
                               num_tx = 4,
                               num_tx_ant = 2)

generate_channel = GenerateOFDMChannel(channel_model = rayleigh,
                                       resource_grid = rg)

apply_channel = ApplyOFDMChannel()
```

where `rg` is an instance of `~sionna.phy.ofdm.ResourceGrid`. Running
the channel model is done as follows:

``` Python
# Generate a batch of channel responses
h = generate_channel(batch_size)
# Apply the channel
# x is the channel input
# no is the noise variance
y = apply_channel([x, h, no])
```

Generating and applying the channel in the time domain can be achieved
by using `~sionna.phy.channel.GenerateTimeChannel` and
`~sionna.phy.channel.ApplyTimeChannel` instead of
`~sionna.phy.channel.GenerateOFDMChannel` and
`~sionna.phy.channel.ApplyOFDMChannel`, respectively.

To access the CIRs, setting up the channel can be done as follows:

``` Python
rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 32,
                               num_tx = 4,
                               num_tx_ant = 2)

apply_channel = ApplyOFDMChannel()
```

and running the channel model as follows:

``` Python
cir = rayleigh(batch_size)
h = cir_to_ofdm_channel(frequencies, *cir)
y = apply_channel([x, h, no])
```

where `frequencies` are the subcarrier frequencies in the baseband,
which can be computed using the
`~sionna.phy.channel.subcarrier_frequencies` utility function.

Applying the channel in the time domain can be done by using
`~sionna.phy.channel.cir_to_time_channel` and
`~sionna.phy.channel.ApplyTimeChannel` instead of
`~sionna.phy.channel.cir_to_ofdm_channel` and
`~sionna.phy.channel.ApplyOFDMChannel`, respectively.

For the purpose of the present document, the following symbols apply:

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 75%" />
</colgroup>
<tbody>
<tr>
<td><span
class="math inline"><em>N</em><sub><em>T</em></sub>(<em>u</em>)</span></td>
<td>Number of transmitters (transmitter index)</td>
</tr>
<tr>
<td><span
class="math inline"><em>N</em><sub><em>R</em></sub>(<em>v</em>)</span></td>
<td>Number of receivers (receiver index)</td>
</tr>
<tr>
<td><span
class="math inline"><em>N</em><sub><em>T</em><em>A</em></sub>(<em>k</em>)</span></td>
<td>Number of antennas per transmitter (transmit antenna index)</td>
</tr>
<tr>
<td><span
class="math inline"><em>N</em><sub><em>R</em><em>A</em></sub>(<em>l</em>)</span></td>
<td>Number of antennas per receiver (receive antenna index)</td>
</tr>
<tr>
<td><span
class="math inline"><em>N</em><sub><em>S</em></sub>(<em>s</em>)</span></td>
<td>Number of OFDM symbols (OFDM symbol index)</td>
</tr>
<tr>
<td><span
class="math inline"><em>N</em><sub><em>F</em></sub>(<em>n</em>)</span></td>
<td>Number of subcarriers (subcarrier index)</td>
</tr>
<tr>
<td><span
class="math inline"><em>N</em><sub><em>B</em></sub>(<em>b</em>)</span></td>
<td>Number of time samples forming the channel input (baseband symbol
index)</td>
</tr>
<tr>
<td><span class="math inline"><em>L</em><sub>min</sub></span></td>
<td>Smallest time-lag for the discrete complex baseband channel</td>
</tr>
<tr>
<td><span class="math inline"><em>L</em><sub>max</sub></span></td>
<td>Largest time-lag for the discrete complex baseband channel</td>
</tr>
<tr>
<td><span class="math inline"><em>M</em>(<em>m</em>)</span></td>
<td>Number of paths (clusters) forming a power delay profile (path
index)</td>
</tr>
<tr>
<td><span
class="math inline"><em>τ</em><sub><em>m</em></sub>(<em>t</em>)</span></td>
<td><span
class="math inline"><em>m</em><sup><em>t</em><em>h</em></sup></span>
path (cluster) delay at time step <span
class="math inline"><em>t</em></span></td>
</tr>
<tr>
<td><span
class="math inline"><em>a</em><sub><em>m</em></sub>(<em>t</em>)</span></td>
<td><span
class="math inline"><em>m</em><sup><em>t</em><em>h</em></sup></span>
path (cluster) complex coefficient at time step <span
class="math inline"><em>t</em></span></td>
</tr>
<tr>
<td><span
class="math inline"><em>Δ</em><sub><em>f</em></sub></span></td>
<td>Subcarrier spacing</td>
</tr>
<tr>
<td><span class="math inline"><em>W</em></span></td>
<td>Bandwidth</td>
</tr>
<tr>
<td><span class="math inline"><em>N</em><sub>0</sub></span></td>
<td>Noise variance</td>
</tr>
</tbody>
</table>

All transmitters are equipped with *N*<sub>*T**A*</sub> antennas and all
receivers with *N*<sub>*R**A*</sub> antennas.

A channel model, such as `~sionna.phy.channel.RayleighBlockFading` or
`~sionna.phy.channel.tr38901.UMi`, is used to generate for each link
between antenna *k* of transmitter *u* and antenna *l* of receiver *v* a
power delay profile
(*a*<sub>*u*, *k*, *v*, *l*, *m*</sub>(*t*), *τ*<sub>*u*, *v*, *m*</sub>), 0 ≤ *m* ≤ *M* − 1.
The delays are assumed not to depend on time *t*, and transmit and
receive antennas *k* and *l*. Such a power delay profile corresponds to
the channel impulse response

$$h\_{u, k, v, l}(t,\tau) =
\sum\_{m=0}^{M-1} a\_{u, k, v, l,m}(t) \delta(\tau - \tau\_{u, v, m})$$

where *δ*(⋅) is the Dirac delta measure. For example, in the case of
Rayleigh block fading, the power delay profiles are time-invariant and
such that for every link (*u*, *k*, *v*, *l*)

$$\begin{aligned}
\begin{align}
M                     &= 1\\
\tau\_{u, v, 0}  &= 0\\
a\_{u, k, v, l, 0}     &\sim \mathcal{CN}(0,1).
\end{align}
\end{aligned}$$

3GPP channel models use the procedure depicted in
<a href="#TR38901" class="citation">[TR38901]</a> to generate power
delay profiles. With these models, the power delay profiles are
time-*variant* in the event of mobility.

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.AWGN

</div>

## Flat-fading channel

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.FlatFadingChannel

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.GenerateFlatFadingChannel

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.ApplyFlatFadingChannel

</div>

<div class="autoclass" members="">

sionna.phy.channel.SpatialCorrelation

</div>

<div class="autoclass" members="">

sionna.phy.channel.KroneckerModel

</div>

<div class="autoclass" members="">

sionna.phy.channel.PerColumnModel

</div>

## Channel model interface

<div class="autoclass" members="">

sionna.phy.channel.ChannelModel

</div>

## Time domain channel

The model of the channel in the time domain assumes pulse shaping and
receive filtering are performed using a conventional sinc filter (see,
e.g., <a href="#Tse" class="citation">[Tse]</a>). Using sinc for
transmit and receive filtering, the discrete-time domain received signal
at time step *b* is

$$y\_{v, l, b} = \sum\_{u=0}^{N\_{T}-1}\sum\_{k=0}^{N\_{TA}-1}
\sum\_{\ell = L\_{\text{min}}}^{L\_{\text{max}}}
\bar{h}\_{u, k, v, l, b, \ell} x\_{u, k, b-\ell}
+ w\_{v, l, b}$$

where *x*<sub>*u*, *k*, *b*</sub> is the baseband symbol transmitted by
transmitter *u* on antenna *k* and at time step *b*,
*w*<sub>*v*, *l*, *b*</sub> ∼ 𝒞𝒩(0, *N*<sub>0</sub>) the additive white
Gaussian noise, and *h̄*<sub>*u*, *k*, *v*, *l*, *b*, *ℓ*</sub> the
channel filter tap at time step *b* and for time-lag *ℓ*, which is given
by

$$\bar{h}\_{u, k, v, l, b, \ell}
= \sum\_{m=0}^{M-1} a\_{u, k, v, l, m}\left(\frac{b}{W}\right)
\text{sinc}\left( \ell - W\tau\_{u, v, m} \right).$$

<div class="note">

<div class="title">

Note

</div>

The two parameters *L*<sub>min</sub> and *L*<sub>max</sub> control the
smallest and largest time-lag for the discrete-time channel model,
respectively. They are set when instantiating
`~sionna.phy.channel.TimeChannel`,
`~sionna.phy.channel.GenerateTimeChannel`, and when calling the utility
function `~sionna.phy.channel.cir_to_time_channel`. Because the sinc
filter is neither time-limited nor causal, the discrete-time channel
model is not causal. Therefore, ideally, one would set
*L*<sub>min</sub> = −∞ and *L*<sub>max</sub> = +∞. In practice, however,
these two parameters need to be set to reasonable finite values. Values
for these two parameters can be computed using the
`~sionna.phy.channel.time_lag_discrete_time_channel` utility function
from a given bandwidth and maximum delay spread. This function returns
−6 for *L*<sub>min</sub>. *L*<sub>max</sub> is computed from the
specified bandwidth and maximum delay spread, which default value is
3*μ**s*. These values for *L*<sub>min</sub> and the maximum delay spread
were found to be valid for all the models available in Sionna when an
RMS delay spread of 100ns is assumed.

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.TimeChannel

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.GenerateTimeChannel

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.ApplyTimeChannel

</div>

<div class="autofunction">

sionna.phy.channel.cir_to_time_channel

</div>

<div class="autofunction">

sionna.phy.channel.time_to_ofdm_channel

</div>

## Channel with OFDM waveform

To implement the channel response assuming an OFDM waveform, it is
assumed that the power delay profiles are invariant over the duration of
an OFDM symbol. Moreover, it is assumed that the duration of the cyclic
prefix (CP) equals at least the maximum delay spread. These assumptions
are common in the literature, as they enable modeling of the channel
transfer function in the frequency domain as a single-tap channel.

For every link (*u*, *k*, *v*, *l*) and resource element (*s*, *n*), the
frequency channel response is obtained by computing the Fourier
transform of the channel response at the subcarrier frequencies, i.e.,

$$\begin{aligned}
\begin{align}
\widehat{h}\_{u, k, v, l, s, n}
&= \int\_{-\infty}^{+\infty} h\_{u, k, v, l}(s,\tau) e^{-j2\pi n \Delta_f \tau} d\tau\\
&= \sum\_{m=0}^{M-1} a\_{u, k, v, l, m}(s)
e^{-j2\pi n \Delta_f \tau\_{u, k, v, l, m}}
\end{align}
\end{aligned}$$

where *s* is used as time step to indicate that the channel response can
change from one OFDM symbol to the next in the event of mobility, even
if it is assumed static over the duration of an OFDM symbol.

For every receive antenna *l* of every receiver *v*, the received signal
*y*<sub>*v*, *l*, *s*, *n*</sub><span class="title-ref"> for resource
element :math:</span>(s, n)\` is computed by

$$y\_{v, l, s, n} = \sum\_{u=0}^{N\_{T}-1}\sum\_{k=0}^{N\_{TA}-1}
\widehat{h}\_{u, k, v, l, s, n} x\_{u, k, s, n}
+ w\_{v, l, s, n}$$

where *x*<sub>*u*, *k*, *s*, *n*</sub> is the baseband symbol
transmitted by transmitter *u*<span class="title-ref"> on antenna
:math:\`k</span> and resource element (*s*, *n*), and
*w*<sub>*v*, *l*, *s*, *n*</sub> ∼ 𝒞𝒩(0, *N*<sub>0</sub>) the additive
white Gaussian noise.

<div class="note">

<div class="title">

Note

</div>

This model does not account for intersymbol interference (ISI) nor
intercarrier interference (ICI). To model the ICI due to channel aging
over the duration of an OFDM symbol or the ISI due to a delay spread
exceeding the CP duration, one would need to simulate the channel in the
time domain. This can be achieved by using the
`~sionna.phy.ofdm.OFDMModulator` and `~sionna.phy.ofdm.OFDMDemodulator`
layers, and the `time domain channel model <time-domain>`. By doing so,
one performs inverse discrete Fourier transform (IDFT) on the
transmitter side and discrete Fourier transform (DFT) on the receiver
side on top of a single-carrier sinc-shaped waveform. This is equivalent
to `simulating the channel in the frequency domain <ofdm-waveform>` if
no ISI nor ICI is assumed, but allows the simulation of these effects in
the event of a non-stationary channel or long delay spreads. Note that
simulating the channel in the time domain is typically significantly
more computationally demanding that simulating the channel in the
frequency domain.

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.OFDMChannel

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.GenerateOFDMChannel

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.ApplyOFDMChannel

</div>

<div class="autofunction">

sionna.phy.channel.cir_to_ofdm_channel

</div>

## Rayleigh block fading

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.RayleighBlockFading

</div>

## 3GPP 38.901 channel models

The submodule `tr38901` implements 3GPP channel models from
<a href="#TR38901" class="citation">[TR38901]</a>.

The `CDL <cdl>`, `UMi <umi>`, `UMa <uma>`, and `RMa <rma>` models
require setting-up antenna models for the transmitters and receivers.
This is achieved using the `~sionna.phy.channel.tr38901.PanelArray`
class.

The `UMi <umi>`, `UMa <uma>`, and `RMa <rma>` models require setting-up
a network topology, specifying, e.g., the user terminals (UTs) and base
stations (BSs) locations, the UTs velocities, etc.
`Utility functions <utility-functions>` are available to help laying out
complex topologies or to quickly setup simple but widely used
topologies.

<div class="autoclass" members="">

sionna.phy.channel.tr38901.PanelArray

</div>

<div class="autoclass" members="">

sionna.phy.channel.tr38901.Antenna

</div>

<div class="autoclass" members="">

sionna.phy.channel.tr38901.AntennaArray

</div>

<div id="tdl">

<div class="autoclass" members="" exclude-members="__call__">

sionna.phy.channel.tr38901.TDL

</div>

</div>

<div id="cdl">

<div class="autoclass" members="" exclude-members="__call__">

sionna.phy.channel.tr38901.CDL

</div>

</div>

<div id="umi">

<div class="autoclass" members="" exclude-members="__call__"
inherited-members="">

sionna.phy.channel.tr38901.UMi

</div>

</div>

<div id="uma">

<div class="autoclass" members="" exclude-members="__call__"
inherited-members="">

sionna.phy.channel.tr38901.UMa

</div>

</div>

<div id="rma">

<div class="autoclass" members="" exclude-members="__call__"
inherited-members="">

sionna.phy.channel.tr38901.RMa

</div>

</div>

## External datasets

<div class="autoclass" members="" exclude-members="__call__"
inherited-members="">

sionna.phy.channel.CIRDataset

</div>

## Utility functions

<div class="autofunction">

sionna.phy.channel.subcarrier_frequencies

</div>

<div class="autofunction">

sionna.phy.channel.time_lag_discrete_time_channel

</div>

<div class="autofunction">

sionna.phy.channel.deg_2_rad

</div>

<div class="autofunction">

sionna.phy.channel.rad_2_deg

</div>

<div class="autofunction">

sionna.phy.channel.wrap_angle_0_360

</div>

<div class="autofunction">

sionna.phy.channel.drop_uts_in_sector

</div>

<div class="autofunction">

sionna.phy.channel.relocate_uts

</div>

<div class="autofunction">

sionna.phy.channel.set_3gpp_scenario_parameters

</div>

<div class="autofunction">

sionna.phy.channel.gen_single_sector_topology

</div>

<div class="autofunction">

sionna.phy.channel.gen_single_sector_topology_interferers

</div>

<div class="autofunction">

sionna.phy.channel.exp_corr_mat

</div>

<div class="autofunction">

sionna.phy.channel.one_ring_corr_mat

</div>

References:  

<div id="citations">

<span id="BHS2017" class="citation-label">BHS2017</span>  
E. Björnson, J. Hoydis, L. Sanguinetti (2017), [“Massive MIMO Networks:
Spectral, Energy, and Hardware
Efficiency”](https://massivemimobook.com), Foundations and Trends in
Signal Processing: Vol. 11, No. 3-4, pp 154–655.

<span id="MAL2018" class="citation-label">MAL2018</span>  
R. K. Mallik, "The exponential correlation matrix: Eigen-analysis and
applications", IEEE Trans. Wireless Commun., vol. 17, no. 7, pp.
4690-4705, Jul. 2018.

<span id="SoS" class="citation-label">SoS</span>  
C. Xiao, Y. R. Zheng and N. C. Beaulieu, "Novel Sum-of-Sinusoids
Simulation Models for Rayleigh and Rician Fading Channels," in IEEE
Transactions on Wireless Communications, vol. 5, no. 12, pp. 3667-3679,
December 2006, doi: 10.1109/TWC.2006.256990.

<span id="TR38901" class="citation-label">TR38901</span>  
3GPP TR 38.901, "Study on channel model for frequencies from 0.5 to 100
GHz", Release 16.1

<span id="TS38141-1" class="citation-label">TS38141-1</span>  
3GPP TS 38.141-1 "Base Station (BS) conformance testing Part 1:
Conducted conformance testing", Release 17

<span id="Tse" class="citation-label">Tse</span>  
D. Tse and P. Viswanath, “Fundamentals of wireless communication“,
Cambridge University Press, 2005.

</div>
