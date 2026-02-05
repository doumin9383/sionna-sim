# Utility Functions

This module provides utility functions for the FEC package. It also
provides serval functions to simplify EXIT analysis of iterative
receivers.

## (Binary) Linear Codes

Several functions are provided to convert parity-check matrices into
generator matrices and vice versa. Please note that currently only
binary codes are supported.

``` Python
# load example parity-check matrix
pcm, k, n, coderate = load_parity_check_examples(pcm_id=3)
```

Note that many research projects provide their parity-check matrices in
the <span class="title-ref">alist</span> format
<a href="#MacKay" class="citation">[MacKay]</a> (e.g., see
<a href="#UniKL" class="citation">[UniKL]</a>). The follwing code
snippet provides an example of how to import an external LDPC
parity-check matrix from an <span class="title-ref">alist</span> file
and how to set-up an encoder/decoder.

``` Python
# load external example parity-check matrix in alist format
al = load_alist(path=filename)
pcm, k, n, coderate = alist2mat(al)

# the linear encoder can be directly initialized with a parity-check matrix
encoder = LinearEncoder(pcm, is_pcm=True)

# initalize BP decoder for the given parity-check matrix
decoder = LDPCBPDecoder(pcm, num_iter=20)

# and run simulation with random information bits
no = 1.
batch_size = 10
num_bits_per_symbol = 2

source = BinarySource()
mapper = Mapper("qam", num_bits_per_symbol)
channel = AWGN()
demapper = Demapper("app", "qam", num_bits_per_symbol)

u = source([batch_size, k])
c = encoder(u)
x = mapper(c)
y = channel(x, no)
llr = demapper(y, no)
c_hat = decoder(llr)
```

<div class="autofunction">

sionna.phy.fec.utils.load_parity_check_examples

</div>

<div class="autofunction">

sionna.phy.fec.utils.alist2mat

</div>

<div class="autofunction">

sionna.phy.fec.utils.load_alist

</div>

<div class="autofunction">

sionna.phy.fec.utils.generate_reg_ldpc

</div>

<div class="autofunction">

sionna.phy.fec.utils.make_systematic

</div>

<div class="autofunction">

sionna.phy.fec.utils.gm2pcm

</div>

<div class="autofunction">

sionna.phy.fec.utils.pcm2gm

</div>

<div class="autofunction">

sionna.phy.fec.utils.verify_gm_pcm

</div>

## EXIT Analysis

The LDPC BP decoder allows to track the internal information flow
(<span class="title-ref">extrinsic information</span>) during decoding
via callbacks. This can be plotted in so-called EXIT Charts
<a href="#tenBrinkEXIT" class="citation">[tenBrinkEXIT]</a> to visualize
the decoding convergence.

![image](../figures/exit.png)

This short code snippet shows how to generate and plot EXIT charts:

``` Python
# parameters
ebno_db = 2.5 # simulation SNR
batch_size = 10000
num_bits_per_symbol = 2 # QPSK
num_iter = 20 # number of decoding iterations

pcm_id = 4 # decide which parity check matrix should be used (0-2: BCH; 3: (3,6)-LDPC 4: LDPC 802.11n
pcm, k, n , coderate = load_parity_check_examples(pcm_id, verbose=True)

noise_var = ebnodb2no(ebno_db=ebno_db,
                      num_bits_per_symbol=num_bits_per_symbol,
                      coderate=coderate)

# init callbacks for tracking of EXIT charts
cb_exit_vn = EXITCallback(num_iter)
cb_exit_cn = EXITCallback(num_iter)

# init components
decoder = LDPCBPDecoder(pcm,
                        hard_out=False,
                        cn_update="boxplus",
                        num_iter=num_iter,
                        v2c_callbacks=[cb_exit_vn,], # register callbacks
                        c2v_callbacks=[cb_exit_cn,],) # register callbacks

# generates fake llrs as if the all-zero codeword was transmitted over an AWNG channel with BPSK modulation
llr_source = GaussianPriorSource()


# generate fake LLRs (Gaussian approximation)
# Remark: the EXIT callbacks require all-zero codeword simulations
llr_ch = llr_source([batch_size, n], noise_var)

# simulate free running decoder (for EXIT trajectory)
decoder(llr_ch)

# calculate analytical EXIT characteristics
# Hint: these curves assume asymptotic code length, i.e., may become inaccurate in the short length regime
Ia, Iev, Iec = get_exit_analytic(pcm, ebno_db)

# and plot the analytical exit curves

plt = plot_exit_chart(Ia, Iev, Iec)

# and add simulated trajectory (requires "track_exit=True")
plot_trajectory(plt, cb_exit_vn.mi.numpy(), cb_exit_cn.mi.numpy(), ebno_db)
```

Remark: for rate-matched 5G LDPC codes, the EXIT approximation becomes
inaccurate due to the rate-matching and the very specific structure of
the code.

<div class="autofunction">

sionna.phy.fec.utils.plot_exit_chart

</div>

<div class="autofunction">

sionna.phy.fec.utils.get_exit_analytic

</div>

<div class="autofunction">

sionna.phy.fec.utils.plot_trajectory

</div>

## Miscellaneous

<div class="autoclass">

sionna.phy.fec.utils.GaussianPriorSource

</div>

<div class="autofunction">

sionna.phy.fec.utils.bin2int

</div>

<div class="autofunction">

sionna.phy.fec.utils.int2bin

</div>

<div class="autofunction">

sionna.phy.fec.utils.bin2int_tf

</div>

<div class="autofunction">

sionna.phy.fec.utils.int2bin_tf

</div>

<div class="autofunction">

sionna.phy.fec.utils.int_mod_2

</div>

<div class="autofunction">

sionna.phy.fec.utils.llr2mi

</div>

<div class="autofunction">

sionna.phy.fec.utils.j_fun

</div>

<div class="autofunction">

sionna.phy.fec.utils.j_fun_inv

</div>

References:  

<div id="citations">

<span id="Brannstrom" class="citation-label">Brannstrom</span>  
F. Brannstrom, L. K. Rasmussen, and A. J. Grant, “Convergence analysis
and optimal scheduling for multiple concatenated codes,” IEEE Trans.
Inform. Theory, vol. 51, no. 9, pp. 3354–3364, 2005.

<span id="Hagenauer" class="citation-label">Hagenauer</span>  
J. Hagenauer, “The Turbo Principle in Mobile Communications,” in Proc.
IEEE Int. Symp. Inf. Theory and its Appl. (ISITA), 2002.

<span id="MacKay" class="citation-label">MacKay</span>  
<http://www.inference.org.uk/mackay/codes/alist.html>

<span id="UniKL" class="citation-label">UniKL</span>  
<https://www.uni-kl.de/en/channel-codes/>

<span id="tenBrink" class="citation-label">tenBrink</span>  
S. ten Brink, G. Kramer, and A. Ashikhmin, “Design of low-density
parity-check codes for modulation and detection,” IEEE Trans. Commun.,
vol. 52, no. 4, pp. 670–678, Apr. 2004.

<span id="tenBrinkEXIT" class="citation-label">tenBrinkEXIT</span>  
S. ten Brink, “Convergence Behavior of Iteratively Decoded Parallel
Concatenated Codes,” IEEE Transactions on Communications, vol. 49, no.
10, pp. 1727-1737, 2001.

</div>
