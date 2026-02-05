# Turbo Codes

This module supports encoding and decoding of Turbo codes
<a href="#Berrou" class="citation">[Berrou]</a>, e.g., as used in the
LTE wireless standard. The convolutional component encoders and decoders
are composed of the `~sionna.phy.fec.conv.encoding.ConvEncoder` and
`~sionna.phy.fec.conv.decoding.BCJRDecoder` layers, respectively.

Please note that various notations are used in literature to represent
the generator polynomials for the underlying convolutional codes. For
simplicity, `~sionna.phy.fec.turbo.encoding.TurboEncoder` only accepts
the binary format, i.e., <span class="title-ref">10011</span>, for the
generator polynomial which corresponds to the polynomial
1 + *D*<sup>3</sup> + *D*<sup>4</sup>.

The following code snippet shows how to set-up a rate-1/3,
constraint-length-4 `~sionna.phy.fec.turbo.encoding.TurboEncoder` and
the corresponding `~sionna.phy.fec.turbo.decoding.TurboDecoder`. You can
find further examples in the [Channel Coding Tutorial
Notebook](../tutorials/5G_Channel_Coding_Polar_vs_LDPC_Codes.html).

Setting-up:

``` Python
encoder = TurboEncoder(constraint_length=4, # Desired constraint length of the polynomials
                       rate=1/3,  # Desired rate of Turbo code
                       terminate=True) # Terminate the constituent convolutional encoders to all-zero state
# or
encoder = TurboEncoder(gen_poly=gen_poly, # Generator polynomials to use in the underlying convolutional encoders
                       rate=1/3, # Rate of the desired Turbo code
                       terminate=False) # Do not terminate the constituent convolutional encoders

# the decoder can be initialized with a reference to the encoder
decoder = TurboDecoder(encoder,
                       num_iter=6, # Number of iterations between component BCJR decoders
                       algorithm="map", # can be also "maxlog"
                       hard_out=True) # hard_decide output
```

Running the encoder / decoder:

``` Python
# --- encoder ---
# u contains the information bits to be encoded and has shape [...,k].
# c contains the turbo encoded codewords and has shape [...,n], where n=k/rate when terminate is False.
c = encoder(u)

# --- decoder ---
# llr contains the log-likelihood ratio values from the de-mapper and has shape [...,n].
# u_hat contains the estimated information bits and has shape [...,k].
u_hat = decoder(llr)
```

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.turbo.TurboEncoder

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.turbo.TurboDecoder

</div>

## Utility Functions

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.turbo.TurboTermination

</div>

<div class="autofunction">

sionna.phy.fec.turbo.utils.polynomial_selector

</div>

<div class="autofunction">

sionna.phy.fec.turbo.utils.puncture_pattern

</div>

References:  

<div id="citations">

<span id="3GPPTS36212_Turbo" class="citation-label">3GPPTS36212_Turbo</span>  
ETSI 3GPP TS 36.212 "Evolved Universal Terrestrial Radio Access (EUTRA);
Multiplexing and channel coding", v.15.3.0, 2018-09.

<span id="Berrou" class="citation-label">Berrou</span>  
C. Berrou, A. Glavieux, P. Thitimajshima, "Near Shannon limit
error-correcting coding and decoding: Turbo-codes", IEEE ICC, 1993.

</div>
