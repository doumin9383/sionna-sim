# Signal

This module contains classes and functions for `filtering <filter>`
(pulse shaping), `windowing <window>`, and `up- <upsampling>` and
`downsampling <downsampling>`. The following figure shows the different
components that can be implemented using this module.

<figure class="align-center">
<img src="../figures/signal_module.png" style="width:75.0%" />
</figure>

This module also contains `utility functions <utility>` for computing
the (inverse) discrete Fourier transform (`FFT <fft>`/`IFFT <ifft>`),
and for empirically computing the
`power spectral density (PSD) <empirical_psd>` and
`adjacent channel leakage ratio (ACLR) <empirical_aclr>` of a signal.

The following code snippet shows how to filter a sequence of QAM
baseband symbols using a root-raised-cosine filter with a Hann window:

``` Python
# Create batch of QAM-16 sequences 
batch_size = 128
num_symbols = 1000
num_bits_per_symbol = 4
x = QAMSource(num_bits_per_symbol)([batch_size, num_symbols])

# Create a root-raised-cosine filter with Hann windowing
beta = 0.22 # Roll-off factor
span_in_symbols = 32 # Filter span in symbols
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
rrcf_hann = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta, window="hann")

# Create instance of the Upsampling layer
us = Upsampling(samples_per_symbol)

# Upsample the baseband x
x_us = us(x)

# Filter the upsampled sequence
x_rrcf = rrcf_hann(x_us)
```

On the receiver side, one would recover the baseband symbols as follows:

``` Python
# Instantiate a downsampling layer
ds = Downsampling(samples_per_symbol, rrcf_hann.length-1, num_symbols)

# Apply the matched filter
x_mf = rrcf_hann(x_rrcf)

# Recover the transmitted symbol sequence
x_hat = ds(x_mf)
```

## Filters

<div class="autoclass"
members="length, window, normalize, coefficients, sampling_times, show, aclr"
exclude-members="call, build">

sionna.phy.signal.SincFilter

</div>

<div class="autoclass"
members="length, window, normalize, coefficients, sampling_times, show, aclr, beta"
exclude-members="call, build">

sionna.phy.signal.RaisedCosineFilter

</div>

<div class="autoclass"
members="length, window, normalize, coefficients, sampling_times, show, aclr, beta"
exclude-members="call, build">

sionna.phy.signal.RootRaisedCosineFilter

</div>

<div class="autoclass"
members="length, window, normalize, coefficients, sampling_times, show, aclr"
exclude-members="call, build">

sionna.phy.signal.CustomFilter

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.signal.Filter

</div>

## Window functions

<div class="autoclass" members="coefficients, length, normalize, show"
exclude-members="call, build">

sionna.phy.signal.HannWindow

</div>

<div class="autoclass" members="coefficients, length, normalize, show"
exclude-members="call, build">

sionna.phy.signal.HammingWindow

</div>

<div class="autoclass" members="coefficients, length, normalize, show"
exclude-members="call, build">

sionna.phy.signal.BlackmanWindow

</div>

<div class="autoclass" members="coefficients, length, normalize, show"
exclude-members="call, build">

sionna.phy.signal.CustomWindow

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.signal.Window

</div>

## Utility Functions

<div class="autofunction">

sionna.phy.signal.convolve

</div>

<div id="fft">

<div class="autofunction">

sionna.phy.signal.fft

</div>

</div>

<div id="ifft">

<div class="autofunction">

sionna.phy.signal.ifft

</div>

</div>

<div id="upsampling">

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.signal.Upsampling

</div>

</div>

<div id="downsampling">

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.signal.Downsampling

</div>

</div>

<div id="empirical_psd">

<div class="autofunction">

sionna.phy.signal.empirical_psd

</div>

</div>

<div id="empirical_aclr">

<div class="autofunction">

sionna.phy.signal.empirical_aclr

</div>

</div>
