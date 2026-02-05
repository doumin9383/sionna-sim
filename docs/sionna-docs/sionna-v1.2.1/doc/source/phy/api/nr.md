# 5G NR

This module provides blocks and functions to support simulations of 5G
NR compliant features, in particular, the physical uplink shared channel
(PUSCH). It provides implementations of a subset of the physical layer
functionalities as described in the 3GPP specifications
<a href="#3GPP38211" class="citation">[3GPP38211]</a>,
<a href="#3GPP38212" class="citation">[3GPP38212]</a>, and
<a href="#3GPP38214" class="citation">[3GPP38214]</a>.

The best way to discover this module's components is by having a look at
the [5G NR PUSCH Tutorial](../tutorials/5G_NR_PUSCH.html).

The following code snippet shows how you can make standard-compliant
simulations of the 5G NR PUSCH with a few lines of code:

``` Python
# Create a PUSCH configuration with default settings
pusch_config = PUSCHConfig()

# Instantiate a PUSCHTransmitter from the PUSCHConfig
pusch_transmitter = PUSCHTransmitter(pusch_config)

# Create a PUSCHReceiver using the PUSCHTransmitter
pusch_receiver = PUSCHReceiver(pusch_transmitter)

# AWGN channel
channel = AWGN()

# Simulate transmissions over the AWGN channel
batch_size = 16
no = 0.1 # Noise variance

x, b = pusch_transmitter(batch_size) # Generate transmit signal and info bits

y = channel(x, no) # Simulate channel output

b_hat = pusch_receiver(y, no) # Recover the info bits

# Compute BER
print("BER:", compute_ber(b, b_hat).numpy())
```

The `~sionna.phy.nr.PUSCHTransmitter` and `~sionna.phy.nr.PUSCHReceiver`
provide high-level abstractions of all required processing blocks. You
can easily modify them according to your needs.

## Carrier

<div class="autoclass" exclude-members="check_config" members="">

sionna.phy.nr.CarrierConfig

</div>

## Layer Mapping

<div class="autoclass" exclude-members="build, call" members="">

sionna.phy.nr.LayerMapper

</div>

<div class="autoclass" exclude-members="build, call" members="">

sionna.phy.nr.LayerDemapper

</div>

## PUSCH

<div class="autoclass"
exclude-members="check_config, l, l_ref, l_0, l_d, l_prime, n"
members="">

sionna.phy.nr.PUSCHConfig

</div>

<div class="autoclass" exclude-members="check_config" members="">

sionna.phy.nr.PUSCHDMRSConfig

</div>

<div class="autoclass" exclude-members="estimate_at_pilot_locations"
members="">

sionna.phy.nr.PUSCHLSChannelEstimator

</div>

<div class="autoclass" inherited-members="">

sionna.phy.nr.PUSCHPilotPattern

</div>

<div class="autoclass" exclude-members="build, call" members="">

sionna.phy.nr.PUSCHPrecoder

</div>

<div class="autoclass" exclude-members="build, call" members="">

sionna.phy.nr.PUSCHReceiver

</div>

<div class="autoclass" exclude-members="build, call" members="">

sionna.phy.nr.PUSCHTransmitter

</div>

## Transport Block

<div class="autoclass" exclude-members="build, call" members="">

sionna.phy.nr.TBConfig

</div>

<div class="autoclass" exclude-members="build, call" members="">

sionna.phy.nr.TBEncoder

</div>

<div class="autoclass" exclude-members="build, call" members="">

sionna.phy.nr.TBDecoder

</div>

## Utils

<div class="autofunction">

sionna.phy.nr.utils.calculate_tb_size

</div>

<div class="autofunction">

sionna.phy.nr.utils.generate_prng_seq

</div>

<div class="autofunction">

sionna.phy.nr.utils.decode_mcs_index

</div>

<div class="autofunction">

sionna.phy.nr.utils.calculate_num_coded_bits

</div>

<div class="autoclass" members="" exclude-members="build, call">

sionna.phy.nr.utils.TransportBlockNR

</div>

<div class="autoclass" members="" exclude-members="build, call">

sionna.phy.nr.utils.CodedAWGNChannelNR

</div>

<div class="autoclass" members="" exclude-members="build, call">

sionna.phy.nr.utils.MCSDecoderNR

</div>

References:  

<div id="citations">

<span id="3GPP38211" class="citation-label">3GPP38211</span>  
3GPP TS 38.211. "NR; Physical channels and modulation."

<span id="3GPP38212" class="citation-label">3GPP38212</span>  
3GPP TS 38.212. "NR; Multiplexing and channel coding"

<span id="3GPP38214" class="citation-label">3GPP38214</span>  
3GPP TS 38.214. "NR; Physical layer procedures for data."

</div>
