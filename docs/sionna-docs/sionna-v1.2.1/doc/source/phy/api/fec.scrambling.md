# Scrambling

The `~sionna.phy.fec.scrambling.Scrambler` module allows to (pseudo)
randomly flip bits in a binary sequence or the signs of a real-valued
sequence, respectively. The `~sionna.phy.fec.scrambling.Descrambler`
implement the corresponding descrambling operation.

To simplify distributed graph execution (e.g., by running scrambler and
descrambler in a different sub-graph/device), the scramblers are
implemented stateless. Thus, the internal seed cannot be update on
runtime and does not change after the initialization. However, if
required an explicit random seed can be passed as additional input the
scrambler/descrambler pair when calling the layer.

Further, the `~sionna.phy.fec.scrambling.TB5GScrambler` enables 5G NR
compliant scrambling as specified in
<a href="#3GPPTS38211_scr" class="citation">[3GPPTS38211_scr]</a>.

The following code snippet shows how to setup and use an instance of the
scrambler:

``` Python
# set-up system
scrambler = Scrambler(seed=1234, # an explicit seed can be provided
                     binary=True) # indicate if bits shall be flipped

descrambler = Descrambler(scrambler=scrambler) # connect scrambler and descrambler

# --- simplified usage with fixed seed ---
# c has arbitrary shape and contains 0s and 1s (otherwise set binary=False)
c_scr = scrambler(c)
# descramble to reconstruct the original order
c_descr = descrambler(c_scr)

# --- advanced usage ---
# provide explicite seed if a new random seed should be used for each call
s = tf.random.uniform((), minval=0, maxval=12345678, dtype=tf.int32)

c_scr = scrambler([c, s])
c_descr = descrambler([c_scr, s])
```

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.scrambling.Scrambler

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.scrambling.TB5GScrambler

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.scrambling.Descrambler

</div>

References:  

<div id="citations">

<span id="3GPPTS38211_scr" class="citation-label">3GPPTS38211_scr</span>  
ETSI 3GPP TS 38.211 "Physical channels and modulation", v.16.2.0,
2020-07.

<span id="Pfister03" class="citation-label">Pfister03</span>  
J. Hou, P. Siegel, L. Milstein, and H. Pfister, "Capacity approaching
bandwidth-efficient coded modulation schemes based on low-density
parity-check codes," IEEE Trans. Inf. Theory, Sep. 2003.

</div>
