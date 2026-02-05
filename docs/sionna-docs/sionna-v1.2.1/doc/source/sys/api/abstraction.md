# PHY Abstraction

<figure class="align-center">
<img src="../figures/phy_abs_api.png" style="width:100.0%" />
</figure>

The physical layer (PHY) abstraction method follows a two-step approach:

1.  The signal-to-interference-plus-noise ratio (SINR) that a single
    codeword experiences across multiple streams, computed via
    `~sionna.phy.ofdm.PostEqualizationSINR`, is aggregated into a single
    *effective* SINR value. The effective SINR is chosen so that, if all
    subcarriers and streams experienced it uniformly, the resulting
    block error rate (BLER) would remain approximately the same.
2.  The effective SINR is then mapped to a BLER value via precomputed
    tables, based on the code block size.

The transport BLER (TBLER) can be finally computed as the probability
that at least one of the code blocks in the transport block is not
correctly received.

For a usage example of PHY abstraction in Sionna, refer to the [Physical
Layer Abstraction notebook](../tutorials/PHY_Abstraction.html).

Next, we formally describe the general principle of effective SINR
mapping (ESM) and the exponential ESM (EESM) model.

We assume the presence of multiple channel "links" *i* = 1, …, *N*, each
characterized by its own SINR<sub>*i*</sub>. In principle, different
codeword symbols can be transmitted on the same link, meaning they
experience the same SINR.\
Let *I*(*x*) measure the "quality" of a link with SINR value *x*, the
exact interpretation of which will be discussed later.\
The effective SINR, SINR<sub>eff</sub>, is defined as the SINR of a
single-link channel whose quality matches the average quality of the
multi-link channel: .. math:: I(mathrm{SINR}\_{text{eff}}) = frac{1}{N}
sum\_{i=1}^N I(mathrm{SINR}\_{i}) .. math:: Rightarrow
mathrm{SINR}\_{text{eff}} = I^{-1} left( frac{1}{N} sum\_{i=1}^N
I(mathrm{SINR}\_{i}) right)\
The form of the quality measure *I* depends on the selected ESM method.\
In the **exponential ESM (EESM)** model, the link quality is defined as:

*I*<sup>EESM</sup>(*x*) := exp (−*x*/*β*).

Thus, the corresponding effective SINR can be expressed as:

<span label="EESM">
$$\mathrm{SINR}\_{\mathrm{eff}}^{\mathrm{EESM}} = -\beta \log \left( \frac{1}{N} \sum\_{i=1}^N e^{-\mathrm{SINR}\_i/\beta} \right).$$
</span>

In the following we outline the derivation of this expression, assuming
the transmission of BPSK (±1) modulated codewords *u*<sup>*A*</sup> and
*u*<sup>*B*</sup>, with a Hamming distance of *d*.

**Single-link channel.** In the basic case with one link (*N* = 1), each
codeword symbol experiences the same channel gain *ρ* and complex noise
power *N*<sub>0</sub>, resulting in the received *real* signal:

$$y^{k}\_j = \sqrt{\rho} u^{k}\_j + w_j, \quad k\in\\A,B\\, \\\forall\\ j$$

where *j* indexes the symbols and
*w*<sub>*j*</sub> ∼ 𝒩(0, *N*<sub>0</sub>/2) is additive real noise.
Hence, the SNR (as well as the SINR, since interference is not
considered) is *ρ*/*N*<sub>0</sub>.\
Codeword *u*<sup>*A*</sup> is incorrectly decoded as *u*<sup>*B*</sup>
when the noise projected along the direction
*u*<sup>*A*</sup> − *u*<sup>*B*</sup> exceeds the half distance between
the two codewords, equal to $\sqrt{d\rho}$. Hence, the pairwise error
probability *P*<sup>*N* = 1</sup>(*u*<sup>*A*</sup> → *u*<sup>*B*</sup>)
can be expressed as:

<span label="pairwise">
$$\begin{aligned}
\begin{align}
    P^{N=1}\left(u^A \rightarrow u^B\right) = & \\ \Pr\left( \xi \sqrt{N_0/2} \>
    \sqrt{d\rho} \right), \quad \xi\sim \mathcal{N}(0,1) \\
    = & \\ Q\left( \sqrt{2 d \\ \mathrm{SINR}} \right) \\
    \le & \\ e^{-d\\ \mathrm{SINR}}
\end{align}
\end{aligned}$$
</span>

where *Q*(*x*) is the tail distribution function of the standard normal
distribution and the inequality stems from the Chernoff bound
*Q*(*x*) ≤ *e*<sup>−*x*<sup>2</sup>/2</sup>, for all *x*.

**Two-link channel.** We now assume that each symbol is transmitted
through channel link 1 or 2 with probabilities *p*<sub>1</sub> and
*p*<sub>2</sub>, respectively. Link *i* = 1, 2 is characterized by its
channel gain $\sqrt{\rho_i}$.\
Consider two received noiseless codewords
*u*<sup>*A*</sup>, *u*<sup>*B*</sup> where *ℓ*<sub>1</sub> = *ℓ* and
*ℓ*<sub>2</sub> = *d* − *ℓ* symbols experience channel 1 and 2,
respectively. Then, their half distance is
$\sqrt{\ell_1 \rho_1 +  \ell_2 \rho_2}$ and the conditioned pairwise
error probability equals:

<span label="conditioned">
$$\begin{aligned}
\begin{align}
    P^{N=2}\left(u^A \rightarrow u^B | \ell_1,\ell_2\right) = & \\ \Pr\left( \sqrt{N_0/2} \xi \> \sqrt{\ell_1 \rho_1 + \ell_2 \rho_2} \right) \\
    = & \\ Q\left( \sqrt{2\ell_1 \\ \mathrm{SINR}\_1 + 2\ell_2 \\ \mathrm{SINR}\_2 } \right).
\end{align}
\end{aligned}$$
</span>

To obtain the pairwise codeword error probability, we average expression
`conditioned` across all (*ℓ*<sub>1</sub>, *ℓ*<sub>2</sub>) events:

<span label="N2">
$$\begin{aligned}
\begin{align}
    P^{N=2}\left(u^A \rightarrow u^B\right) & \\ = \sum\_{\ell=0}^d {d \choose \ell} p_1^{\ell} \\ p_2^{d-\ell} \\ P^{N=2}\left(u^A \rightarrow u^B | \ell_1=\ell,\ell_2=d-\ell\right) \\
    \le & \\ \sum\_{\ell=0}^d {d \choose \ell} \left(p_1 e^{-\mathrm{SINR}\_1}\right)^\ell \left( p_2 e^{-\mathrm{SINR}\_2} \right)^{d-\ell} \\
    = & \\ \left( p_1 e^{-\mathrm{SINR}\_1}  + p_2 e^{-\mathrm{SINR}\_2}\right)^d
\end{align}
\end{aligned}$$
</span>

where the inequality stems again from the Chernoff bound.

**Multi-link channel.** Expression `N2` extends to a multi-link channel
(*N* ≥ 2) as follows:

<span label="pr_multistate">
$$\begin{align}
    P^{N}\left(u^A \rightarrow u^B\right) \le \\ \left( \sum\_{i=1}^N p_i e^{-\mathrm{SINR}\_i} \right)^d.
\end{align}$$
</span>

**EESM expression.** By equating the multi-link pairwise error
probability bound `pr_multistate` with the analogous single-link
expression `pairwise`, we recognize that the multi-link channel is
analogous to a single-link channel with SINR:

$$\mathrm{SINR}\_{\mathrm{eff}}^{\mathrm{EESM}} := -\log \left( \sum\_{i=1}^N p_i e^{-\mathrm{SINR}\_i} \right)$$

where SINR<sub>eff</sub><sup>EESM</sup> is the *effective* SINR for the
multi-link channel under the EESM model.\
If we further assume that all links are equiprobable, i.e.,
*p*<sub>*i*</sub> = 1/*N* for all *i*, then we obtain expression `EESM`
with *β* = 1.

Note that the introduction of parameter *β* in `EESM` is useful to adapt
the EESM formula to different modulation and coding schemes (MCS), since
the argument above holds for BPSK modulation only. Hence, *β* shall
depend on the used MCS, as shown in
<a href="#5GLENA" class="citation">[5GLENA]</a>.

<div class="autoclass" members="" exclude-members="call, build">

sionna.sys.EffectiveSINR

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.sys.EESM

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.sys.PHYAbstraction

</div>

References:  

<div id="citations">

<span id="5GLENA" class="citation-label">5GLENA</span>  
S. Lagen, K. Wanuga, H. Elkotby, S. Goyal, N. Patriciello, L. Giupponi.
["New radio physical layer abstraction for system-level simulations of
5G networks"](https://arxiv.org/abs/2001.10309). IEEE International
Conference on Communications (ICC), 2020

</div>
