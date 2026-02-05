# Polar Codes

The Polar code module supports 5G-compliant Polar codes and includes
successive cancellation (SC), successive cancellation list (SCL), and
belief propagation (BP) decoding.

The module supports rate-matching and CRC-aided decoding. Further,
Reed-Muller (RM) code design is available and can be used in combination
with the Polar encoding/decoding algorithms.

The following code snippets show how to setup and run a rate-matched 5G
compliant Polar encoder and a corresponding successive cancellation list
(SCL) decoder.

First, we need to create instances of
`~sionna.phy.fec.polar.encoding.Polar5GEncoder` and
`~sionna.phy.fec.polar.decoding.Polar5GDecoder`:

``` Python
encoder = Polar5GEncoder(k          = 100, # number of information bits (input)
                         n          = 200) # number of codeword bits (output)


decoder = Polar5GDecoder(encoder    = encoder, # connect the Polar decoder to the encoder
                         dec_type   = "SCL", # can be also "SC" or "BP"
                         list_size  = 8)
```

Now, the encoder and decoder can be used by:

``` Python
# --- encoder ---
# u contains the information bits to be encoded and has shape [...,k].
# c contains the polar encoded codewords and has shape [...,n].
c = encoder(u)

# --- decoder ---
# llr contains the log-likelihood ratios from the demapper and has shape [...,n].
# u_hat contains the estimated information bits and has shape [...,k].
u_hat = decoder(llr)
```

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.polar.encoding.Polar5GEncoder

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.polar.encoding.PolarEncoder

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.polar.decoding.Polar5GDecoder

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.polar.decoding.PolarSCDecoder

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.polar.decoding.PolarSCLDecoder

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.polar.decoding.PolarBPDecoder

</div>

## Utility Functions

<div class="autofunction">

sionna.phy.fec.polar.utils.generate_5g_ranking

</div>

<div class="autofunction">

sionna.phy.fec.polar.utils.generate_polar_transform_mat

</div>

<div class="autofunction">

sionna.phy.fec.polar.utils.generate_rm_code

</div>

<div class="autofunction">

sionna.phy.fec.polar.utils.generate_dense_polar

</div>

References:  

<div id="citations">

<span id="3GPPTS38212" class="citation-label">3GPPTS38212</span>  
ETSI 3GPP TS 38.212 "5G NR Multiplexing and channel coding", v.16.5.0,
2021-03.

<span id="Arikan_BP" class="citation-label">Arikan_BP</span>  
E. Arikan, “A Performance Comparison of Polar Codes and Reed-Muller
Codes,” IEEE Commun. Lett., vol. 12, no. 6, pp. 447-449, Jun. 2008.

<span id="Arikan_Polar" class="citation-label">Arikan_Polar</span>  
E. Arikan, "Channel polarization: A method for constructing
capacity-achieving codes for symmetric binary-input memoryless
channels," IEEE Trans. on Information Theory, 2009.

<span id="Bioglio_Design" class="citation-label">Bioglio_Design</span>  
V. Bioglio, C. Condo, I. Land, "Design of Polar Codes in 5G New Radio,"
IEEE Communications Surveys & Tutorials, 2020. Online availabe
<https://arxiv.org/pdf/1804.04389.pdf>

<span id="Cammerer_Hybrid_SCL" class="citation-label">Cammerer_Hybrid_SCL</span>  
Sebastian Cammerer, Benedikt Leible, Matthias Stahl, Jakob Hoydis, and
Stephan ten Brink, "Combining Belief Propagation and Successive
Cancellation List Decoding of Polar Codes on a GPU Platform," IEEE
ICASSP, 2017.

<span id="Ebada_Design" class="citation-label">Ebada_Design</span>  
M. Ebada, S. Cammerer, A. Elkelesh and S. ten Brink, “Deep
Learning-based Polar Code Design”, Annual Allerton Conference on
Communication, Control, and Computing, 2019.

<span id="Forney_Graphs" class="citation-label">Forney_Graphs</span>  
G. D. Forney, “Codes on graphs: normal realizations,” IEEE Trans.
Inform. Theory, vol. 47, no. 2, pp. 520-548, Feb. 2001.

<span id="Goala_LP" class="citation-label">Goala_LP</span>  
N. Goela, S. Korada, M. Gastpar, "On LP decoding of Polar Codes," IEEE
ITW 2010.

<span id="Gross_Fast_SCL" class="citation-label">Gross_Fast_SCL</span>  
Seyyed Ali Hashemi, Carlo Condo, and Warren J. Gross, "Fast and Flexible
Successive-cancellation List Decoders for Polar Codes." IEEE Trans. on
Signal Processing, 2017.

<span id="Hashemi_SSCL" class="citation-label">Hashemi_SSCL</span>  
Seyyed Ali Hashemi, Carlo Condo, and Warren J. Gross, "Simplified
Successive-Cancellation List Decoding of Polar Codes." IEEE ISIT, 2016.

<span id="Hui_ChannelCoding" class="citation-label">Hui_ChannelCoding</span>  
D. Hui, S. Sandberg, Y. Blankenship, M. Andersson, L. Grosjean "Channel
coding in 5G new radio: A Tutorial Overview and Performance Comparison
with 4G LTE," IEEE Vehicular Technology Magazine, 2018.

<span id="Stimming_LLR" class="citation-label">Stimming_LLR</span>  
Alexios Balatsoukas-Stimming, Mani Bastani Parizi, Andreas Burg,
"LLR-Based Successive Cancellation List Decoding of Polar Codes." IEEE
Trans Signal Processing, 2015.

<span id="Tal_SCL" class="citation-label">Tal_SCL</span>  
Ido Tal and Alexander Vardy, "List Decoding of Polar Codes." IEEE Trans
Inf Theory, 2015.

</div>
