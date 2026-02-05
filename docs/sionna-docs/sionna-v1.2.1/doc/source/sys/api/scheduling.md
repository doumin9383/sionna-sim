# Scheduling

<figure class="align-center">
<img src="../figures/scheduling_api.png" style="width:100.0%" />
</figure>

Since the spectrum is shared among multiple users, resources must be
allocated in a fair and efficient manner. On the one hand, it is
desirable to allocate resources uniformly across users. On the other
hand, in the presence of fading, it is crucial to schedule users when
their channel conditions are favorable.

The proportional fairness (PF) scheduler achieves both objectives by
maximizing the sum of logarithms of the long-term throughputs *T*(*u*)
across the users *u* = 1, 2, …:

max ∑<sub>*u*</sub>log *T*(*u*).

For a usage example of user scheduling in Sionna, refer to the
[Proportional Fairness Scheduler
notebook](../tutorials/Scheduling.html).

<div class="autoclass" members="" exclude-members="call, build">

sionna.sys.PFSchedulerSUMIMO

</div>

References:

> 

<div id="citations">

<span id="Jalali00" class="citation-label">Jalali00</span>  
A. Jalali, R. Padovani, R. Pankaj, "Data throughput of CDMA-HDR a high
efficiency-high data rate personal communication wireless system."
VTC2000-Spring. 2000 IEEE 51st Vehicular Technology Conference
Proceedings. Vol. 3. IEEE, 2000.

</div>
