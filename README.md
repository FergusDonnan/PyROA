# PyROA

## Overview
PyROA is a tool for modelling quasar lightcurves ([Donnan et al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210712318D/abstract)) where the variability is described using a running optimal average (ROA), and paramters are sampled using Markov Chain Monte Carlo (MCMC) techniques - specifically using emcee. Using a Bayesian approach, priors can be used on the sampled parameters. It can also intercalibrate same-filter ligthcurves from different telescopes and adjust their uncertainties accordingly ([Donnan et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230209370D/abstract)).
Currently it has three main uses:

1. Determining the time delay between lightcurves at different wavelengths. 
2. Intercalibrating lightcurves from multiple telescopes, merging them into a single lightcurve.
3. Determining the time delay between images of lensed quasars, where the microlensing effects are also modelled.

PyROA also includes a noise model, where there is a parameter for each lightcurve that adds extra variance to the flux measurments, to account for underestimated errors. This can be turned off if required.

The code is easy to use with example jupyter notebooks provided, demonstrating each of the three main uses. The code is ran by specifying a directory which contains each of the lightcurves as a .dat file with the columns being time, flux, flux_err in that order. All that is also needed is to specify the prior, specifically the limits of a uniform prior on each of the parameters.

If you have any queries, email me at: fergus.donnan@physics.ox.ac.uk

## Installation

Install using pip: pip install PyROA


## Changes

#### v3.1.0
- Added new utility functions (Utils.py) to analyse the outputs of PyROA. See the Utils_Tutorial for examples of usage. These include displaying lightcurve(+residuals), Convergence plots, corner and chain plots by parameter, Lag spectrum plot and Flux-Flux analysis plot.
- Changes to the intercalibration files were added. It now outputs, for every datum the original telescope, ROA model+uncertainty at the point, and degree of freedom value.

#### v3.0.0
- Re-implemented the delay distribution function to now use a numerical convolution. This allows any type of transfer function to be modelled.
- Added use of the emcee backend, which allows saving and resuming of progress.


#### v2.0.2
- Fixed bug in calculating the slow component.

#### v2.0.1
- Fixed bug in driving lightcurve output when using the delay distributions.

#### v2.0.0

- Added option to Fit() to specify which filter is the reference to measure delays relative to. Specify the name of the filter to the parameter delay_ref.

- Changed the delay distribution to a truncated Gaussian which prevents delays less than the delay of the blurring reference lightcurve contributing to the delay distribution. 

- If including a slow component, the code should now run at the same speed as when not including the component.

#### v1.0.3
Initial release of PyROA

## Usage
### Case 1: Measuring delays between lightcurves of different wavelengths
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

<strong> class Fit(datadir, objName, filters, priors, init_tau = None, init_delta=1.0, delay_dist=False, add_var=True, sig_level = 4.0, Nsamples=10000, Nburnin=5000, include_slow_comp=False, slow_comp_delta=30.0, calc_P=False) </strong>

<strong> Parameters: 
    
datadir : string :</strong> Directory of lightcurves in format "objName_filter.dat". 

<strong> objName : string :</strong> Name of object in order to find lightcurve .dat files 

<strong> filters : array :</strong> List of filter names.

<strong> priors : array :</strong> Array specifying the limits of uniform priors for the parameters. Exact formatting is explained above.

<strong> init_tau : array :</strong> List of initial time delays. This can help reduce burn-in or find correct solution if delays are very large and lightcurves have little overlap.

<strong> init_delta : float :</strong> Initial values of delta.

<strong> delay_dist : bool :</strong> Whether to include a delay distribution for each time delay that blurs the lightcurve according to the width of the delay distribution. If set to True, each delay parameter now represents the peak of a truncated Gaussian delay distribtuion and a new parameter, tau_rms, represents the width. This is blurring measured relative to the first lightcurve, where the trucation point of the distribtuion is the delay of this lightcurve, preventing delays less than this band contributing to the delay distribution. This option does take longer to run, especially on larger data sets.

<strong> add_var : bool :</strong> Whether to include paramters that add extra variance to the flux errors for each lightcurve. 

<strong> sig_level : float :</strong> The threshold in units of sigma, for the sigma clipping.

<strong> Nsamples : int :</strong> The number of MCMC samples, per walker, for the fitting procedure. This value includes the burn-in 

<strong> Nburnin : int :</strong> The number of Nsamples to be removed as burn-in. 

<strong> include_slow_comp : bool :</strong> Whether to include a slow varying component to the model, represented by a ROA with a fixed wide window function specified by slow_comp_delta.

<strong> include_slow_comp : int :</strong> Width of the window function for the slow component if included. 

<strong> calc_P : bool :</strong> Option to pre-calculate the number of parameters for the ROA as a function of delta, which is subsequently interpolated for use in the fitting routine. This option can increase run-time significantly for large data sets. WARNING: This is approximate as it does not account for the current time delay or extra variance parameters. Would recomend only using if delays are small and add_var = False.

<strong> delay_ref : string :</strong> Name of the filter, matching the filters array, that the delays will be measured relative to. This is default set to the first filter provided.


### Case 2: Intercalibrating lightcurves from multiple telescopes

When using data from multiple telescopes e.g from the Las Cumbres Observatory, these can be combined into a single lightcurve where the running optimal average provides a model of the merged lightcurve. 
Similar to before this is ran by specifying a directory which contains each lightcurve as a .dat file with three columns: time, flux, flux_err. The files must be named: "objName_filter_scope.dat". Therefore to run the code after specifying the directory, provide the objName, the filter of the merged lightcurve, and a list of telescopes that are to be merged:

```` import PyROA
datadir = "/F9_lightcurves/"
objName="F9"
filter="B"
    
    
#List of telescope names
scopes=["1m003", "1m004", "1m005", "1m009", "1m010", "1m011", "1m012", "1m013"]
#Priors
priors = [[0.01, 10.0], [0.0, 2.0]]

fit = PyROA.InterCalibrate(datadir, objName, filter, scopes, priors) 
````

Here the priors are just for delta and the extra error parameters in the form: priors = [[delta_lower, delta_upper], [sig_lower, sig_upper]].

This outputs the merged lightcurve as a .dat file to the same directory as the one specified. A corner plot is also created from where the function is ran.
The full list of options for the InterCalibrate function are:

<strong> class InterCalibrate(datadir, objName, filter, scopes, priors, init_delta=1.0, sig_level = 4.0, Nsamples=15000, Nburnin=10000) </strong>

<strong> Parameters: 
    
datadir : string :</strong> Directory of lightcurves in format "objName_filter_scope.dat"

<strong> objName : string :</strong> Name of object in order to find lightcurve .dat files 

<strong> filter : string :</strong> Name of filter of merged lightcurve.

<strong> scopes : array :</strong> List of telescope names.

<strong> priors : array :</strong> Array specifying the limits of uniform priors for the parameters. Exact formatting is explained above.

<strong> init_delta : float :</strong> Initial values of delta.

<strong> sig_level : float :</strong> The threshold in units of sigma, for the sigma clipping.

<strong> Nsamples : int :</strong> The number of MCMC samples, per walker, for the fitting procedure. This value includes the burn-in 

<strong> Nburnin : int :</strong> The number of Nsamples to be removed as burn-in. 






### Case 3: Measuring time delays between lensed quasar images
To measure the time delay between lightcurves of lensed quasar images we use the function GravLensFit. This is ran in the same way as before where a directory is specified that contains .dat files of each of the lightcurves with three columns: time, mag, mag_err. Here the brightness is in magnitude and the function does the conversion where : flux = 3.0128e-5 10^(-0.4m). This converts into arbitrary flux units and so this factor can be changed depending on the data.
Here we specify images rather than filters:

```` import PyROA
datadir = "/PG 1115+080/"
objName="PG 1115+080"
images=["A","B","C"]

priors = [[0.0, 5.0],[0.0, 50.0], [-400.0, 400.0], [2.5, 150.0], [0.0, 2.0], [-50.0, 50.0]]

fit = PyROA.GravLensFit(datadir, objName, images, priors, init_delta=10.0, Nsamples=20000, Nburnin=15000)
````
Here the priors are for:
priors = [[A1_lower, A1_upper], [B1_lower, B1_upper], [tau_lower, tau_upper],[delta_lower, delta_upper], [sig_lower, sig_upper], [P_lower, P_upper]], where the A1 and B1 are the rms and mean of the first lightcurve in the arbitrary flux units, tau is the time delays between images, delta is the ROA window width, sig is the extra variance parameter and P is the prior range for all the microlensing polynomial coefficients.




<strong> class GravLensFit(datadir, objName, images, priors, init_tau = None, init_delta=10.0, add_var=True, sig_level = 4.0, Nsamples=10000, Nburnin=5000, flux_convert_factor=3.0128e-5) </strong>

<strong> Parameters: 
    
datadir : string :</strong> Directory of lightcurves in format "objName_image.dat"

<strong> objName : string :</strong> Name of object in order to find lightcurve .dat files 

<strong> images : array :</strong> List of images.

<strong> priors : array :</strong> Array specifying the limits of uniform priors for the parameters. Exact formatting is explained above.

<strong> init_tau : array :</strong> List of initial time delays. This can help reduce burn-in or find correct solution if delays are very large and lightcurves have little overlap.

<strong> init_delta : float :</strong> Initial values of delta.

<strong> add_var : bool :</strong> Whether to include paramters that add extra variance to the flux errors for each lightcurve.

<strong> sig_level : float :</strong> The threshold in units of sigma, for the sigma clipping.

<strong> Nsamples : int :</strong> The number of MCMC samples, per walker, for the fitting procedure. This value includes the burn-in 

<strong> Nburnin : int :</strong> The number of Nsamples to be removed as burn-in. 

<strong> flux_convert_factor : float :</strong> Factor used when converting magnitudes to fluxes, where flux = flux_convert_factor * 10^(-0.4m).

## Citation

If you use this code please cite ([Donnan et al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210712318D/abstract)):
````
@ARTICLE{2021MNRAS.508.5449D,
       author = {{Donnan}, Fergus R. and {Horne}, Keith and {Hern{\'a}ndez Santisteban}, Juan V.},
        title = "{Bayesian analysis of quasar light curves with a running optimal average: new time delay measurements of COSMOGRAIL gravitationally lensed quasars}",
      journal = {\mnras},
     keywords = {gravitational lensing: strong, methods: data analysis, quasars: general, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2021,
        month = dec,
       volume = {508},
       number = {4},
        pages = {5449-5467},
          doi = {10.1093/mnras/stab2832},
archivePrefix = {arXiv},
       eprint = {2107.12318},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.5449D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
````
The intercalibration procedure is found at [Donnan et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230209370D/abstract):
````
@ARTICLE{2023arXiv230209370D,
       author = {{Donnan}, Fergus R. and {Hern{\'a}ndez Santisteban}, Juan V. and {Horne}, Keith and {Hu}, Chen and {Du}, Pu and {Li}, Yan-Rong and {Xiao}, Ming and {Ho}, Luis C. and {Aceituno}, Jes{\'u}s and {Wang}, Jian-Min and {Guo}, Wei-Jian and {Yang}, Sen and {Jiang}, Bo-Wei and {Yao}, Zhu-Heng},
        title = "{Testing Super-Eddington Accretion onto a Supermassive Black Hole: Reverberation Mapping of PG 1119+120}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2023,
        month = feb,
          eid = {arXiv:2302.09370},
        pages = {arXiv:2302.09370},
          doi = {10.48550/arXiv.2302.09370},
archivePrefix = {arXiv},
       eprint = {2302.09370},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230209370D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
````


