# Cyclic Redundancy Check (CRC)

A cyclic redundancy check adds parity bits to detect transmission
errors. The following code snippets show how to add CRC parity bits to a
bit sequence and how to verify that the check is valid.

First, we need to create instances of `~sionna.phy.fec.crc.CRCEncoder`
and `~sionna.phy.fec.crc.CRCDecoder`:

``` Python
encoder = CRCEncoder(crc_degree="CRC24A") # the crc_degree denotes the number of added parity bits and is taken from the 3GPP 5G NR standard.

decoder = CRCDecoder(crc_encoder=encoder) # the decoder must be associated to a specific encoder
```

We can now run the CRC encoder and test if the CRC holds:

``` Python
# u contains the information bits to be encoded and has shape [...,k].
# c contains u and the CRC parity bits. It has shape [...,k+k_crc].
c = encoder(u)

# u_hat contains the information bits without parity bits and has shape [...,k].
# crc_valid contains a boolean per codeword that indicates if the CRC validation was successful.
# It has shape [...,1].
u_hat, crc_valid = decoder(c)
```

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.crc.CRCEncoder

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.fec.crc.CRCDecoder

</div>

References:  

<div id="citations">

<span id="3GPPTS38212_CRC" class="citation-label">3GPPTS38212_CRC</span>  
ETSI 3GPP TS 38.212 "5G NR Multiplexing and channel coding", v.16.5.0,
2021-03.

</div>
