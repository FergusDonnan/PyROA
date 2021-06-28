# PyROA

PyROA is a tool for modelling quasar lightcurves where the variability is described using a running optimal average (ROA), and paramters are sampled using Markov Chain Monte Carlo (MCMC) techniques - specifically using emcee. Using a Bayesian approach, priors can be used on the sampled parameters.
Currently it has three main uses:

1. Determining the time delay between lightcurves at different wavelengths. 
2. Intercalibrating lightcurves from multiple telescopes, merging them into a single lightcurve.
3. Determining the time delay between images of lensed quasars, where the microlensing effects are also modelled.

PyROA also includes a noise model, where there is a parameter for each lightcurve that adds extra variance to the flux measurments, to account for underestimated errors. This can be turned of if required.

The code is easy to use with example jupyter notebooks provided, demonstrating each of the three main uses. The code is ran by specifying a directory which contains each of the lightcurves as a .dat file with the columns being time, flux, flux_err in that order. All that is also needed is to specify the prior, specifically the limits of a uniform prior on each of the parameters.





