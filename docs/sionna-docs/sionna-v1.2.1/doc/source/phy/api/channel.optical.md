# Optical Channel Models

This module provides layers and functions that implement channel models
for (fiber) optical communications. The currently only available model
is the split-step Fourier method (`~sionna.phy.channel.SSFM`, for dual-
and single-polarization) that can be combined with an Erbium-doped
amplifier (`~sionna.phy.channel.EDFA`).

The following code snippets show how to setup and simulate the
transmission over a single-mode fiber (SMF) by using the split-step
Fourier method.

``` Python
# init fiber
span = sionna.phy.channel.optical.SSFM(
                              alpha=0.046,
                              beta_2=-21.67,
                              f_c=193.55e12,
                              gamma=1.27,
                              length=80,
                              n_ssfm=200,
                              n_sp=1.0,
                              t_norm=1e-12,
                              with_amplification=False,
                              with_attenuation=True,
                              with_dispersion=True,
                              with_nonlinearity=True,
                              dtype=tf.complex64)
# init amplifier
amplifier = sionna.phy.channel.optical.EDFA(
                              g=4.0,
                              f=2.0,
                              f_c=193.55e12,
                              dt=1.0e-12)

@tf.function
def simulate_transmission(x, n_span):
      y = x
      # simulate n_span fiber spans
      for _ in range(n_span):
            # simulate single span
            y = span(y)
            # simulate amplifier
            y = amplifier(y)

      return y
```

Running the channel model is done as follows:

``` Python
# x is the optical input signal, n_span the number of spans
y = simulate_transmission(x, n_span)
```

For further details, the tutorial ["Optical Channel with Lumped
Amplification"](../tutorials/Optical_Lumped_Amplification_Channel.html)
provides more sophisticated examples of how to use this module.

For the purpose of the present document, the following symbols apply:

<table>
<colgroup>
<col style="width: 30%" />
<col style="width: 69%" />
</colgroup>
<tbody>
<tr>
<td><span class="math inline"><em>T</em><sub>norm</sub></span></td>
<td>Time normalization for the SSFM in <span
class="math inline">(s)</span></td>
</tr>
<tr>
<td><span class="math inline"><em>L</em><sub>norm</sub></span></td>
<td>Distance normalization the for SSFM in <span
class="math inline">(m)</span></td>
</tr>
<tr>
<td><span class="math inline"><em>W</em></span></td>
<td>Bandwidth</td>
</tr>
<tr>
<td><span class="math inline"><em>α</em></span></td>
<td>Attenuation coefficient in <span
class="math inline">(1/<em>L</em><sub>norm</sub>)</span></td>
</tr>
<tr>
<td><span class="math inline"><em>β</em><sub>2</sub></span></td>
<td>Group velocity dispersion coeff. in <span
class="math inline">(<em>T</em><sub>norm</sub><sup>2</sup>/<em>L</em><sub>norm</sub>)</span></td>
</tr>
<tr>
<td><span class="math inline"><em>f</em><sub>c</sub></span></td>
<td>Carrier frequency in <span class="math inline">(Hz)</span></td>
</tr>
<tr>
<td><span class="math inline"><em>γ</em></span></td>
<td>Nonlinearity coefficient in <span
class="math inline">(1/<em>L</em><sub>norm</sub>/W)</span></td>
</tr>
<tr>
<td><span class="math inline"><em>ℓ</em></span></td>
<td>Fiber length in <span
class="math inline">(<em>L</em><sub>norm</sub>)</span></td>
</tr>
<tr>
<td><span class="math inline"><em>h</em></span></td>
<td>Planck constant</td>
</tr>
<tr>
<td><span class="math inline"><em>N</em><sub>SSFM</sub></span></td>
<td>Number of SSFM simulation steps</td>
</tr>
<tr>
<td><span class="math inline"><em>n</em><sub>sp</sub></span></td>
<td>Spontaneous emission factor of Raman amplification</td>
</tr>
<tr>
<td><span
class="math inline"><em>Δ</em><sub><em>t</em></sub></span></td>
<td>Normalized simulation time step in <span
class="math inline">(<em>T</em><sub>norm</sub>)</span></td>
</tr>
<tr>
<td><span
class="math inline"><em>Δ</em><sub><em>z</em></sub></span></td>
<td>Normalized simulation step size in <span
class="math inline">(<em>L</em><sub>norm</sub>)</span></td>
</tr>
<tr>
<td><span class="math inline"><em>G</em></span></td>
<td>Amplifier gain</td>
</tr>
<tr>
<td><span class="math inline"><em>F</em></span></td>
<td>Amplifier's noise figure</td>
</tr>
<tr>
<td><span class="math inline"><em>ρ</em><sub>ASE</sub></span></td>
<td>Noise spectral density</td>
</tr>
<tr>
<td><span class="math inline"><em>P</em></span></td>
<td>Signal power</td>
</tr>
<tr>
<td><span class="math inline"><em>D̂</em></span></td>
<td>Linear SSFM operator <a href="#A2012"
class="citation">[A2012]</a></td>
</tr>
<tr>
<td><span class="math inline"><em>N̂</em></span></td>
<td>Non-linear SSFM operator <a href="#A2012"
class="citation">[A2012]</a></td>
</tr>
<tr>
<td><span class="math inline"><em>f</em><sub>sim</sub></span></td>
<td>Simulation bandwidth</td>
</tr>
</tbody>
</table>

**Remark:** Depending on the exact simulation parameters, the SSFM
algorithm may require `dtype=tf.complex128` for accurate simulation
results. However, this may increase the simulation complexity
significantly.

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.SSFM

</div>

<div class="autoclass" members="" exclude-members="call, build">

sionna.phy.channel.EDFA

</div>

## Utility functions

<div class="autofunction">

sionna.phy.channel.utils.time_frequency_vector

</div>

References:  

<div id="citations">

<span id="A2012" class="citation-label">A2012</span>  
G. P. Agrawal, "Fiber-optic Communication Systems", 4th ed. Wiley Series
in Microwave and Optical Engineering 222. New York: Wiley, 2010.

<span id="BGT2000" class="citation-label">BGT2000</span>  
D. M. Baney, P. Gallion, and R. S. Tucker, "Theory and Measurement
Techniques for the Noise Figure of Optical Amplifiers", Optical Fiber
Technology 6, No. 2, 2000.

<span id="EKWFG2010" class="citation-label">EKWFG2010</span>  
R. J. Essiambre, G. Kramer, P. J. Winzer, G. J. Foschini, and B. Goebel,
"Capacity Limits of Optical Fiber Networks", Journal of Lightwave
Technology 28, No. 4, 2010.

<span id="FMF1976" class="citation-label">FMF1976</span>  
J. A. Fleck, J. R. Morris, and M. D. Feit, "Time-dependent Propagation
of High Energy Laser Beams Through the Atmosphere", Appl. Phys., Vol.
10, pp 129–160, 1976.

<span id="GD1991" class="citation-label">GD1991</span>  
C. R. Giles, and E. Desurvire, "Modeling Erbium-Doped Fiber Amplifiers",
Journal of Lightwave Technology 9, No. 2, 1991.

<span id="HT1973" class="citation-label">HT1973</span>  
R. H. Hardin and F. D. Tappert, "Applications of the Split-Step Fourier
Method to the Numerical Solution of Nonlinear and Variable Coefficient
Wave Equations.", SIAM Review Chronicles, Vol. 15, No. 2, Part 1, p 423,
1973.

<span id="MFFP2009" class="citation-label">MFFP2009</span>  
N. J. Muga, M. C. Fugihara, M. F. S. Ferreira, and A. N. Pinto, "ASE
Noise Simulation in Raman Amplification Systems", Conftele, 2009.

<span id="WMC1991" class="citation-label">WMC1991</span>  
P. K. A. Wai, C. R. Menyuk, and H. H. Chen, "Stability of Solitons in
Randomly Varying Birefringent Fibers", Optics Letters, No. 16, 1991.

</div>
