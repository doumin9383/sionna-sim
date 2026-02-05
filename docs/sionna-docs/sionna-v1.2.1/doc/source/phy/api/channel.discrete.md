# Discrete Channel Models

This module provides layers and functions that implement channel models
with discrete input/output alphabets.

All channel models support binary inputs *x* ∈ {0, 1} and
<span class="title-ref">bipolar</span> inputs *x* ∈ {−1, 1},
respectively. In the later case, it is assumed that each
<span class="title-ref">0</span> is mapped to
<span class="title-ref">-1</span>.

The channels can either return discrete values or log-likelihood ratios
(LLRs). These LLRs describe the channel transition probabilities
*L*(*y*|*X* = 1) = *L*(*X* = 1|*y*) + *L*<sub>*a*</sub>(*X* = 1) where
$L_a(X=1)=\operatorname{log} \frac{P(X=1)}{P(X=0)}$ depends only on the
<span class="title-ref">a priori</span> probability of *X* = 1. These
LLRs equal the <span class="title-ref">a posteriori</span> probability
if *P*(*X* = 1) = *P*(*X* = 0) = 0.5.

Further, the channel reliability parameter *p*<sub>*b*</sub> can be
either a scalar value or a tensor of any shape that can be broadcasted
to the input. This allows for the efficient implementation of channels
with non-uniform error probabilities.

The channel models are based on the
<span class="title-ref">Gumble-softmax trick</span>
<a href="#GumbleSoftmax" class="citation">[GumbleSoftmax]</a> to ensure
differentiability of the channel w.r.t. to the channel reliability
parameter. Please see
<a href="#LearningShaping" class="citation">[LearningShaping]</a> for
further details.

Setting-up:

\>\>\> bsc = BinarySymmetricChannel(return_llrs=False,
bipolar_input=False)

Running:

\>\>\> x = tf.zeros((128,)) \# x is the channel input \>\>\> pb = 0.1 \#
pb is the bit flipping probability \>\>\> y = bsc((x, pb))

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.BinaryErasureChannel

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.BinaryMemorylessChannel

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.BinarySymmetricChannel

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.BinaryZChannel

</div>

References:  

<div id="citations">

<span id="GumbleSoftmax" class="citation-label">GumbleSoftmax</span>  
E. Jang, G. Shixiang, and B. Poole. <span class="title-ref">"Categorical
reparameterization with gumbel-softmax,"</span> arXiv preprint
arXiv:1611.01144 (2016).

<span id="LearningShaping" class="citation-label">LearningShaping</span>  
M. Stark, F. Ait Aoudia, and J. Hoydis. <span class="title-ref">"Joint
learning of geometric and probabilistic constellation shaping,"</span>
2019 IEEE Globecom Workshops (GC Wkshps). IEEE, 2019.

</div>
