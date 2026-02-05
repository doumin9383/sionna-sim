# Linear Codes

This package provides generic support for binary linear block codes.

For encoding, a universal `~sionna.phy.fec.linear.LinearEncoder` is
available and can be initialized with either a generator or parity-check
matrix. The matrix must be binary and of full rank.

For decoding, `~sionna.phy.fec.linear.OSDecoder` implements the
ordered-statistics decoding (OSD) algorithm
<a href="#Fossorier" class="citation">[Fossorier]</a> which provides
close to maximum-likelihood (ML) estimates for a sufficiently large
order of the decoder. Please note that OSD is highly complex and not
feasible for all code lengths.

*Remark:* As this package provides support for generic encoding and
decoding (including Polar and LDPC codes), it cannot rely on code
specific optimizations. To benefit from an optimized decoder and keep
the complexity as low as possible, please use the code specific
enc-/decoders whenever available.

The encoder and decoder can be set up as follows:

``` Python
pcm, k, n, coderate = load_parity_check_examples(pcm_id=1) # load example code

# or directly import an external parity-check matrix in alist format
al = load_alist(path=filename)
pcm, k, n, coderate = alist2mat(al)

# encoder can be directly initialized with the parity-check matrix
encoder = LinearEncoder(enc_mat=pcm, is_pcm=True)

# decoder can be initialized with generator or parity-check matrix
decoder = OSDecoder(pcm, t=4, is_pcm=True) # t is the OSD order

# or instantiated from a specific encoder
decoder = OSDecoder(encoder=encoder, t=4) # t is the OSD order
```

We can now run the encoder and decoder:

``` Python
# u contains the information bits to be encoded and has shape [...,k].
# c contains codeword bits and has shape [...,n]
c = encoder(u)

# after transmission LLRs must be calculated with a demapper
# let's assume the resulting llr_ch has shape [...,n]
c_hat = decoder(llr_ch)
```

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.linear.LinearEncoder

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.linear.OSDecoder

</div>

References:  

<div id="citations">

<span id="Fossorier" class="citation-label">Fossorier</span>  
M. Fossorier, S. Lin, "Soft-Decision Decoding of Linear Block Codes
Based on Ordered Statistics", IEEE Trans. Inf. Theory, vol. 41, no.5,
1995.

<span id="Stimming_LLR_OSD" class="citation-label">Stimming_LLR_OSD</span>  
A.Balatsoukas-Stimming, M. Parizi, A. Burg, "LLR-Based Successive
Cancellation List Decoding of Polar Codes." IEEE Trans Signal
Processing, 2015.

</div>
