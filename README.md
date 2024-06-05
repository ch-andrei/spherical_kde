Note: Modified by Andrei Chubarau from https://github.com/williamjameshandley/spherical_kde

Spherical Kernel Density Estimation
===================================

These packages allow you to do rudimentary kernel density estimation on a
sphere. Suggestions for improvements/extensions welcome.

The fundamental principle is to replace the traditional Gaussian function used
in 
[kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation)
with the
[Von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution).

Bandwidth estimation is still rough-and-ready.
