# System Level (SYS)

This package provides differentiable system-level simulation
functionalities for multi-cell networks.

It is based on a `physical layer abstraction <api/abstraction>` that
computes the block error rate (BLER) from the
`~sionna.phy.ofdm.PostEqualizationSINR`. It further includes Layer-2
functionalities, such as `link adaption (LA)<api/link_adaptation>` for
adaptive modulation and coding scheme (MCS) selection, downlink and
uplink `power control<api/power_control>`, and
`user scheduling<api/scheduling>`. Base stations can be placed on a
`spiral hexagonal<api/topology>` grid, where wraparound is used for
pathloss computation.

<figure class="align-center">
<img src="figures/sionna_sys.png" style="width:100.0%" />
</figure>

A good starting point for Sionna SYS is the available
`tutorials <tutorials>` page.

<div class="toctree" hidden="" maxdepth="6">

tutorials api/sys.rst

</div>
