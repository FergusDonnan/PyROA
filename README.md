# PyROA

## Overview
PyROA is a tool for modelling quasar lightcurves where the variability is described using a running optimal average (ROA), and paramters are sampled using Markov Chain Monte Carlo (MCMC) techniques - specifically using emcee. Using a Bayesian approach, priors can be used on the sampled parameters.
Currently it has three main uses:

1. Determining the time delay between lightcurves at different wavelengths. 
2. Intercalibrating lightcurves from multiple telescopes, merging them into a single lightcurve.
3. Determining the time delay between images of lensed quasars, where the microlensing effects are also modelled.

PyROA also includes a noise model, where there is a parameter for each lightcurve that adds extra variance to the flux measurments, to account for underestimated errors. This can be turned of if required.

The code is easy to use with example jupyter notebooks provided, demonstrating each of the three main uses. The code is ran by specifying a directory which contains each of the lightcurves as a .dat file with the columns being time, flux, flux_err in that order. All that is also needed is to specify the prior, specifically the limits of a uniform prior on each of the parameters.



## Installation







## Usage
#### Case 1: Measuring delays between lightcurves of different wavelengths
To measure the time delay between lightcurves of different wavelengths we first specify a directory, object name and filters. In the directory each lightcurve is a .dat file that contains three columns: time, flux, flux_err and is named: "objName_filter.dat". Using the mock data provided we would run the code using

```` import PyROA
datadir = "/MockData/HighSN/"
objName="TestObj"
filters=["1","2","3"]
    
priors = [[0.0, 20.0],[0.0, 100.0], [-50.0, 50.0], [0.01, 10.0], [0.0, 10.0]]

fit = PyROA.Fit(datadir, objName, filters, priors, add_var=True)

````

Priors are uniform where the limits are specified in the following way:

priors = [[A_lower, A_upper], [B_lower, B_upper], [tau_lower, tau_upper],[delta_lower, delta_upper], [sig_lower, sig_upper]]. In the above these limits are large but here is a quick rundown of what they mean:
- The first parameter here is A, which the rms of each lightcurve (there is an A1, A2, A3 for three lightcurves) and must be positive hence some limits between 0 and some large value are appropriate. A1, A2, A3 share the same uniform prior. 
- The next parameter, B, represents the mean of each lightcurve (there is an B1, B2, B3 for three lightcurves). 
- Next, tau, is the time delay between lightcurves (here there is only tau2, tau3) and so this prior range gives the range of lags explored by the model. 
- The next parameter delta gives the width of the window function which must be positive and non-zero. If your probability is returning nan, it may be because the lower limit on this prior is too small. 
- The final parameter is the extra error parameter, which again is positive.

Per lightcurve there are 4 parameters: A, B, tau, sig, with tau=0 for the first lightcurve.
Delta controls the flexability of the running optimal average which is calculated using all of the lightcurves. 

There are many more options that can be specified when using Fit. A full explanation is below:




#### Case 2: Intercalibrating lightcurves from multiple telescopes

When using data from multiple telescopes e.g from the Las Cumbres Observatory, these can be combined into a single lightcurve where the running optimal average provides a model of the merged lightcurve. 



#### Case 3: Measuring time delays between lensed quasar images

## Citation


