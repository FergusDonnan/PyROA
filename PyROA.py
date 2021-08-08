import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import interpolate
import emcee
from tqdm import tqdm
from multiprocessing import Pool
from itertools import chain
from tabulate import tabulate
import corner
from astropy.modeling import models
import os
import pickle
import numba
from numba import jit
from numba import prange
from numba.typed import List
from numba import generated_jit, types
from scipy import special
import scipy.special
import matplotlib
from astropy.modeling import models
from scipy import signal






@jit(nopython=True, cache=True, parallel=True)
def RunningOptimalAverage(t_data, Flux, Flux_err, delta):
    #Inputs
    # Flux : Array of data values
    # Flux_err : Array containig errors of data values
    # delta : parameter defining how "loose" memory function is
    # t_data : Array of wavelength data values

    
    #Outputs
    # t : List of model times 
    # model : List of model fluxes calculated from running optimal average


    gridsize=1000

    
    mx=max(t_data)
    mn=min(t_data)
    length = abs(mx-mn)
    t = np.arange(mn, mx, length/(gridsize))
    model = np.empty(gridsize)
    errs = np.empty(gridsize)
     
    for j in prange(len(t)):

        #Only include significant data points
        t_data_use = t_data[np.where(np.absolute(t[j]-t_data) < 5.0*delta)[0]]
        Flux_err_use = Flux_err[np.where(np.absolute(t[j]-t_data) < 5.0*delta)[0]]
        Flux_use = Flux[np.where(np.absolute(t[j]-t_data) < 5.0*delta)[0]]
        

        if (len(np.where(np.absolute(t[j]-t_data) < 5.0*delta)[0])<1):
            #Define Gaussian Memory Function
            w =  np.exp(-0.5*(((t[j]-t_data)/delta)**2))/(Flux_err**2)
        
            #1/cosh Memory Function
            #w = 1.0/((Flux_err**2)*np.cosh((t[j]-t_data)/delta))
            
            #Lorentzian
            #w = 1.0/((Flux_err**2)*(1.0+((t[j]-t_data)/delta)**2))
          
            #Boxcar
            #w=np.full(len(Flux_err), 0.01) # zero
          

            
            w_sum = np.nansum(w)
            #To avoid NAN, 
            if (w_sum==0):
                model[j] = model[j-1]
                errs[j] = errs[j-1]
            else:
                #Calculate optimal average
                model[j] = np.nansum(Flux*w)/w_sum
                #Calculate error
                errs[j] = np.sqrt(1.0/w_sum) 


        else:
            #Define Gaussian Memory Function
            w =np.exp(-0.5*(((t[j]-t_data_use)/delta)**2))/(Flux_err_use**2)
        
            #1/cosh Memory Function
            #w = 1.0/((Flux_err_use**2)*np.cosh((t[j]-t_data_use)/delta))
            
            #Lorentzian
           # w = 1.0/((Flux_err_use**2)*(1.0+((t[j]-t_data_use)/delta)**2))
            
            #Boxcar
            #w = 1.0/(Flux_err_use**2)
            w_sum = np.nansum(w)
            #Calculate optimal average
            model[j] = np.nansum(Flux_use*w)/w_sum
            #Calculate error
            errs[j] = np.sqrt(1.0/w_sum)
        



    return t[0:int(gridsize)], model, errs



def Con(delta, rms, a):
    bounds = np.sqrt(delta**2 + rms**2)
    x = np.linspace(0 - 5.0*bounds, 10.0*bounds, 200)
    del2 = delta**2
    rms2 = rms**2
    a2 = a**2
    x2 = x**2
    absx = np.abs(x)
    d = np.sqrt(del2 + rms2)
    d2 = del2 + rms2
    rp2 = np.sqrt(np.pi/2.0)
    r2 = 1.0/np.sqrt(2.0)
    
    B = np.sqrt((1.0/del2) + (1.0/rms2))
       
    E = (-1.0*rp2*delta*B*rms2*special.erf(r2*x/(B*rms2))) + (rp2*rms*d*special.erf(x*delta/np.sqrt(2.*del2*rms2 + 2.*(rms2**2)))) +(rms*(rp2*delta*rms*B +rp2*d*special.erf(r2*((x*rms2) - (a*d2))/(delta*rms*d))))
      

    con = E*np.exp(-0.5*x2/d2)*delta/d2 # Trunc. Gaussian
    #con = np.exp(-0.5*x2/d2) # Gaussian
    
    
    return con/np.max(con), x, d
    
    
    

def CalcWinds(t_data, Flux, Flux_err, delta, rmss, N, sizes,  taus):




    
    conv = np.empty((len(t_data),200))
    factors = np.empty((len(t_data),len(t_data)))
    
    ts = np.empty((len(t_data),200))
    ds = np.empty(len(t_data))
    for i in range(int(N)):
        #Define delay distribution
        l=int(np.sum(sizes[0:i+1]))
        u=int(np.sum(sizes[0:i+2]))




        #Needs generalised
        factors[l:u, :] = np.sqrt(delta**2 + rmss[l]**2)/np.sqrt(delta**2 + rmss[l]**2+ (rmss[l]-rmss)**2)
        
        #Calculate convolution
        cutoff = taus[0] - taus[l]
        c, t, d = Con(delta, rmss[l], cutoff)
        conv[l:u, :] = c
        ts[l:u, :] = t
        ds[l:u] = d

    return factors, conv, ts, ds




@jit(nopython=True, cache=True, parallel=True)
def RunningOptimalAverageConv(t_data, Flux, Flux_err, deltas, factors, conv, t):
    #Inputs
    # Flux : Array of data values
    # Flux_err : Array containig errors of data values
    # delta : parameter defining how "loose" memory function is
    # t_data : Array of wavelength data values

    
    #Outputs
    # t : List of model times 
    # model : List of model fluxes calculated from running optimal average
    

    

    mjd=t_data
    model = np.empty(len(mjd))
    errs = np.empty(len(mjd))
    Ps = np.empty(len(mjd))
    
    for j in prange(len(mjd)):

        delta = deltas[j]
        #Retrieve convolutions
        convl = conv[j,:]
        #Shift to each data point
        t_shift = t[j,:] + t_data[j]
        conv_full = np.interp(t_data, t_shift , convl) 


        #Only include significant data points
        factors_full = factors[j,:]
       

        factors_full_use = factors_full[np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0]]
        
        
        conv_use = conv_full[np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0]]
        Flux_err_use = Flux_err[np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0]]
        Flux_use = Flux[np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0]]
        if (np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0].size<1):
            #Define Gaussian Memory Function
            w = factors_full  * conv_full/(Flux_err**2)
        
            #1/cosh Memory Function
            #w = 1.0/((Flux_err_use**2)*np.cosh((t[j]-t_data_use)/delta))
        
        
            w_sum = np.nansum(w)
            #To avoid NAN, 
            if (w_sum==0):
                model[j] = model[j-1]
                errs[j] = errs[j-1]
            else:
                #Calculate optimal average
                model[j] = np.nansum(Flux*w)/w_sum
                #Calculate error
                errs[j] = np.sqrt(1.0/w_sum)  


        else:
            #Define Gaussian Memory Function
            w =factors_full_use* conv_use/(Flux_err_use**2)
        
            #1/cosh Memory Function
            #w = 1.0/((Flux_err_use**2)*np.cosh((t[j]-t_data_use)/delta))
        
            w_sum = np.nansum(w)
            #Calculate optimal average
            model[j] = np.nansum(Flux_use*w)/w_sum
            #Calculate error
            errs[j] = np.sqrt(1.0/w_sum)
        
        Ps[j] = 1.0/((Flux_err[j]**2)*w_sum)
    P = np.nansum(Ps)
    return mjd, model, errs, P








@jit(nopython=True, cache=True, parallel=True)
def RunningOptimalAverageOutConv(mjd, t_data, Flux, Flux_err, factors, conv, prev, t, delta):
    #Inputs
    # Flux : Array of data values
    # Flux_err : Array containig errors of data values
    # delta : parameter defining how "loose" memory function is
    # t_data : Array of wavelength data values

    
    #Outputs
    # t : List of model times 
    # model : List of model fluxes calculated from running optimal average



        #Retrieve convolutions
    convl = conv[prev,:]



    factors_full = factors[prev,:]
        
        
        
    model = np.empty(len(mjd))
    errs = np.empty(len(mjd))
     
    for j in prange(len(mjd)):
    
        #Shift to each data point
        conv = np.interp(t_data, t[prev,:] + mjd[j] , convl)    
    
            
        #Only include significant data points
        factors_full_use = factors_full[np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0]]
        conv_use = conv[np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0]]  
        Flux_err_use = Flux_err[np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0]]
        Flux_use = Flux[np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0]]    
    
        if (len(np.where(np.absolute(mjd[j]-t_data) < 5.0*delta)[0])<1):
            #Define Gaussian Memory Function
            w = factors_full  * conv/(Flux_err**2)
        
            #1/cosh Memory Function
            #w = 1.0/((Flux_err_use**2)*np.cosh((t[j]-t_data_use)/delta))
            
            w_sum = np.nansum(w)
            #To avoid NAN, 
            if (w_sum==0):
                model[j] = model[j-1]
                errs[j] = errs[j-1]
            else:
                #Calculate optimal average
                model[j] = np.nansum(Flux*w)/w_sum
                #Calculate error
                errs[j] = np.sqrt(1.0/w_sum) 


        else:
            #Define Gaussian Memory Function
            w =factors_full_use* conv_use/(Flux_err_use**2)
        
            #1/cosh Memory Function
            #w = 1.0/((Flux_err_use**2)*np.cosh((t[j]-t_data_use)/delta))
        
            #Calculate optimal average
            model[j] = np.nansum(Flux_use*w)/np.nansum(w)
            #Calculate error
            errs[j] = np.sqrt(1.0/np.nansum(w))

    return mjd, model, errs















@jit(nopython=True, cache=True, parallel=True)
def CalculateP(t_data, Flux, Flux_err, delta):

    Ps = np.empty(len(t_data))
    for i in prange(len(t_data)):
    
    
        #Only include significant data points
        t_data_use = t_data[np.where(np.absolute(t_data[i]-t_data) < 5.0*delta)[0]]
        Flux_err_use = Flux_err[np.where(np.absolute(t_data[i]-t_data) < 5.0*delta)[0]]

        
        if (len(np.where(np.absolute(t_data[i]-t_data) < 5.0*delta)[0])==0):
            #Define Gaussian Memory Function
            w =np.exp(-0.5*(((t_data[i]-t_data)/delta)**2))/(Flux_err**2)

            #1/cosh Memory function
            #w = 1.0/((Flux_err**2)*np.cosh((t_data[i]-t_data)/delta))

            #Lorentzian
           # w = 1.0/((Flux_err**2)*(1.0+((t_data[i]-t_data)/delta)**2))
           
           #Boxcar
            #w=np.full(len(Flux_err), 0.01)

            
            
        else:
        
            #Define Gaussian Memory Function
            w =np.exp(-0.5*(((t_data[i]-t_data_use)/delta)**2))/(Flux_err_use**2)

            #1/cosh Memory function
            #w = 1.0/((Flux_err_use**2)*np.cosh((t_data[i]-t_data_use)/delta))

            #Lorentzian
            #w = 1.0/((Flux_err_use**2)*(1.0+((t_data[i]-t_data_use)/delta)**2))
            
            #Boxcar
            #w=1.0/(Flux_err_use**2)
        w_sum = np.nansum(w)

        #P= P + 1.0/((Flux_err[i]**2)*np.nansum(w))
        if (w_sum==0):
            w_sum = 1e-300
        Ps[i] = 1.0/((Flux_err[i]**2)*w_sum)

    return np.nansum(Ps)
    

    
    

    

    
    

    
    
    
        
############################################### Echo Mapping Model ####################################################




    
#BIC
def BIC(params, data, add_var, size, sig_level,include_slow_comp, slow_comp_delta, P_func, slow_comps, P_slow, init_delta, delay_dist, pos_ref):


    Nchunk = 3
    if (add_var == True):
        Nchunk +=1
    if (delay_dist == True):
        Nchunk+=1
        param_delete=2
    else:
        param_delete=1
        

    Npar =  Nchunk*len(data) + 1
        
    chunk_size = int((Npar - 1)/len(data))

    #Break params list into chunks of 3 i.e A, B, tau in each chunk
    params_chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size )] 
    
    #Extract delta and extra variance parameters as last in params list
    if (delay_dist == True):
        delta = params_chunks[-1][0]
        rmss = np.zeros(size)
        taus = np.zeros(size)
    
    else:
        delta = params_chunks[-1][0]

    #Loop through each lightcurve and shift data by parameters
    merged_mjd = np.zeros(size)
    merged_flux = np.zeros(size)
    merged_err = np.zeros(size)

    prev=0
    sizes = np.zeros(int(len(data)+1))
    for i in range(len(data)):
        A = params_chunks[i][0]
        B = params_chunks[i][1]   
        tau = params_chunks[i][2]
        if (add_var == True):
            V =  params_chunks[i][-1]
            
        if (delay_dist == True):
            if (i>0):
                tau_rms = params_chunks[i][3]
            else:
                tau_rms=delta/100.0


            

        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        sizes[i+1] = len(mjd)
        

        #Add extra variance
        if (add_var == True):
            err = np.sqrt((err**2) + (V**2))
        

                   
        if (include_slow_comp==True):
            t_slow, m_slow, errs_slow = slow_comps[i]


            flux = (flux - B - np.interp(mjd, slow_comps[i][0], slow_comps[i][1]))/A
        else:
            flux = (flux - B )/A   
            P_slow[i]=0.0 
            
        err = err/A
        

            #Shift data
        mjd = mjd - tau


        #Add shifted data to merged lightcurve
        for j in range(len(mjd)):
            merged_mjd[int(j+ prev)] = mjd[j]
            merged_flux[int(j+ prev)] = flux[j]
            merged_err[int(j+ prev)] = err[j]
            if (delay_dist == True):
                rmss[int(j+ prev)] = tau_rms
                taus[int(j+ prev)] = tau
               # factors[int(j+ prev)] = delta/np.sqrt(delta**2 + (tau_rms)**2)#/delta
     
        prev = int(prev + len(mjd))
        



    #Calculate ROA to merged lc
    if (delay_dist == False):
        t, m, errs = RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
        
        #Normalise lightcurve

        m_mean = np.mean(m)#np.average(m, weights = 1.0/(errs**2))

        m_rms = np.std(m)
        m = (m-m_mean)/m_rms
        errs = errs/m_rms
        
        
        #Calculate no. of parameters
        if (P_func == None):
                                              
            P=CalculateP(merged_mjd, merged_flux, merged_err, delta)
        else:
            P = P_func(delta)
         
    #Calculate no. of paramters for delay_dist==True here, actual ROA calcualted in loop per lightcurve   
    else:
        factors, conv, x, d = CalcWinds(merged_mjd, merged_flux, merged_err, delta, rmss, len(data), sizes,  taus)
        t,m,errs, P = RunningOptimalAverageConv(merged_mjd, merged_flux, merged_err, d, factors, conv, x) 




    #Calculate chi-squared for each lightcurve and sum
    lps=[0]*len(data)
    prev=0
    for i in range(len(data)):

        A = params_chunks[i][0]
        B = params_chunks[i][1] 
        tau = params_chunks[i][2] 
        if (add_var == True):
            V =  params_chunks[i][-1]

        #Originial lightcurves
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        #Add extra variance
        if (add_var == True):
            err = np.sqrt((err**2) + (V**2)) 
            



        if (delay_dist == False):
            t_shifted = t + tau
            #interp = interpolate.interp1d(t_shifted, m, kind="linear", fill_value="extrapolate")
            m_m = m#interp(t)
        

                    
            if (include_slow_comp==True):
                m_s = np.interp(t_shifted, slow_comps[i][0], slow_comps[i][1])  #Not sure - originally t not t_shifted
                m_scaled = A*(m_m) + B + m_s
            else:
                m_scaled = A*(m_m) + B

         
            #Model
            interp = interpolate.interp1d(t_shifted, m_scaled, kind="linear", fill_value="extrapolate")
            model = interp(mjd)
            
        #Calculate ROA at mjd of each lightcurve using different delta
        else:
            if (i>0):
                tau_rms =params_chunks[i][3]
            else:
                tau_rms=delta/100.0
           # delta_new = np.sqrt(delta**2 + (tau_rms)**2)
          #  factor = delta/np.sqrt(delta**2 + (tau_rms)**2)
            
            
            
            
            m_mean = np.mean(m[prev : int(prev + len(mjd))])
            m_rms = np.std(m[prev : int(prev + len(mjd))])
            m[prev : int(prev + len(mjd))] = (m[prev : int(prev + len(mjd))]-m_mean)/m_rms
            errs[prev : int(prev + len(mjd))] = errs[prev : int(prev + len(mjd))]/m_rms  
             
            Xs = m[prev : int(prev + len(mjd))]
            errs = errs[prev : int(prev + len(mjd))]
            
            
            if (include_slow_comp==True):
                m_s = np.interp(mjd, slow_comps[i][0], slow_comps[i][1])
                
                model = A*Xs + B + m_s
            else:
                model = A*Xs + B
                     
        prev = int(prev + len(mjd))
            

        chi2 = np.empty(len(mjd))
        ex_term = np.empty(len(mjd))  
        for j in range(len(mjd)):

            if(abs(model[j]-flux[j]) < sig_level*err[j]):
            
            
                chi2[j] = ((model[j]-flux[j])**2)/(err[j]**2)
                
                ex_term[j] = np.log(((err[j]**2)/(data[i][j,2]**2)))  
                              
            else:
                chi2[j] =sig_level**2
                ex_term[j] = np.log(((abs(model[j] - flux[j])/sig_level)**2)/(data[i][j,2]**2))
        lps[i]=np.sum(chi2 + ex_term) 
    
    lprob = np.sum(lps)  
    
    #Calculate Penalty
    Penalty = 0.0
    for i in range(len(data)):
        mjd = data[i][:,0]
        
        if (i==pos_ref):

            Penalty = Penalty + float(chunk_size+P_slow[i] - 1.0)*np.log(len(mjd))
        else:
            Penalty = Penalty + float(chunk_size+P_slow[i])*np.log(len(mjd))        
            
    Penalty = Penalty + (P*np.log(len(merged_flux)))
        
    BIC =  lprob + Penalty


    if (math.isnan(BIC) == True):
        return -np.inf
    else:
        return BIC
    
    
    
    
 
#Priors
def log_prior(params, priors, add_var, data, delay_dist):
    Nchunk = 3
    if (add_var == True):
        Nchunk +=1
    if (delay_dist == True):
        Nchunk+=1

        
    Npar =  Nchunk*len(data) + 1
    
    
    chunk_size = int((Npar - 1)/len(data))


    #Break params list into chunks of 3 i.e A, B, tau in each chunk
    params_chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size)]
    
    #Extract delta and extra variance parameters as last in params list
    delta = params_chunks[-1][0]

        
    #Read in priors
    A_prior = priors[0]
    B_prior = priors[1]
    tau_prior = priors[2]
    delta_prior = priors[3]
    if (add_var == True): 
        V_prior = priors[4]
    
    if (delay_dist == True):
        rms_prior_width = priors[5]

    
    check=[]
    #V_priors=np.empty(len(data))
    #Loop over lightcurves
    pr=[]
    for i in range(len(data)):
        A = params_chunks[i][0]
        B = params_chunks[i][1]
        tau = params_chunks[i][2]
        
        if (add_var == True):
            V =  params_chunks[i][-1]
            
            
        if (delay_dist == True and i>0):
            if (params_chunks[i][3]>=0.0):
                tau_rms = params_chunks[i][3]
                pr.append(2.0*np.log((1.0/np.sqrt(2.0*np.pi*(rms_prior_width**2)))*np.exp(-0.5*(tau_rms/rms_prior_width)**2)))
            else:
                check.append(1.0)
        else:
            pr.append(0.0)
            
        #Force peak delays to be larger than blurring reference
        if (delay_dist == True):
            if (tau >=params_chunks[0][2]):
                check.append(0.0)
            else:
                check.append(1.0)


             
        if (add_var == True):
            if A_prior[0] <= A <= A_prior[1] and B_prior[0] <= B <= B_prior[1] and tau_prior[0] <= tau <= tau_prior[1] and V_prior[0]<= V <= V_prior[1]:
                check.append(0.0)
            else:
                check.append(1.0)
        else:
            if A_prior[0] <= A <= A_prior[1] and B_prior[0] <= B <= B_prior[1] and tau_prior[0] <= tau <= tau_prior[1]:
                check.append(0.0)
            else:
                check.append(1.0)

               
                    
            
    if np.sum(np.array(check)) == 0.0 and delta_prior[0]<= delta <= delta_prior[1]:
        return 0.0 + np.sum(pr)
    else:
        return -np.inf

    

    
    
#Probability
def log_probability(params, data, priors, add_var, size, sig_level, include_slow_comp, slow_comp_delta,P_func, slow_comps, P_slow, init_delta, delay_dist, pos_ref):


    #Insert t1 as zero for syntax


    Nchunk = 3
    if (add_var == True):
        Nchunk +=1
    if (delay_dist == True):
        Nchunk+=1
        if (pos_ref == 0):
            params=np.insert(params, [2], [0.0])    #Insert zero for reference delay dist
        else:
            params=np.insert(params, [3], [0.0])    #Insert zero for reference delay dist
                 
    Npar =  Nchunk*len(data) + 1
    pos = pos_ref*Nchunk + 2
    params=np.insert(params, pos, [0.0])    #Insert zero for reference delay 
    
    
    
    
    lp = log_prior(params, priors, add_var, data, delay_dist)
    if not np.isfinite(lp):
        return -np.inf
    return lp - BIC(params, data, add_var, size, sig_level, include_slow_comp, slow_comp_delta,P_func, slow_comps, P_slow, init_delta, delay_dist, pos_ref)

    
    






     

def FullFit(data, priors, init_tau, init_delta, add_var, sig_level, Nsamples, Nburnin, include_slow_comp, slow_comp_delta, calc_P, delay_dist, pos_ref):

    Nchunk = 3
    if (add_var == True):
        Nchunk +=1
    if (delay_dist == True):
        Nchunk+=1
        param_delete=2
    else:
        param_delete=1
        

    Npar =  Nchunk*len(data) + 1    

    ########################################################################################    
    #Run MCMC to fit to data
    
    #Choose intial conditions from mean and rms of data
    pos = [0]*Npar
    labels = [None]*Npar
    chunk_size = int((Npar - 1)/len(data))
    
    pos_chunks = [pos[i:i + chunk_size] for i in range(0, len(pos), chunk_size)]
    labels_chunks = [labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)]
    
    size = 0
    merged_mjd = []
    merged_flux = []
    merged_err = []
    sizes = np.zeros(int(len(data)+1))
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        size = size + len(mjd)
        sizes[i+1] = len(mjd)   
     
        if (include_slow_comp==True):
            t_slow, m_slow, errs_slow = RunningOptimalAverage(mjd,flux,err, slow_comp_delta)
            m_slow = m_slow - np.mean(m_slow)
            m_s = interpolate.interp1d(t_slow, m_slow, kind="linear", fill_value="extrapolate")
            pos_chunks[i][0] = np.std(flux - m_s(mjd)) #Set intial A to rms of data
            pos_chunks[i][1] = np.mean(flux- m_s(mjd)) #Set initial B to mean of data
        else:        
            pos_chunks[i][0] = pos_chunks[i][0] + np.std(flux)# - m_s(mjd)) #Set intial A to rms of data
            pos_chunks[i][1] = np.mean(flux)#- m_s(mjd)) #Set initial B to mean of data
            
            
        pos_chunks[i][2] = init_tau[i]
        

        

        if(add_var == True):
            pos_chunks[i][-1] =  0.01 #Set initial V
            labels_chunks[i][-1] = "\u03C3"+str(i)
            
            
        if (delay_dist == True):
            pos_chunks[i][3] = 1.0
            labels_chunks[i][3]="\u0394"+str(i)
                       
        labels_chunks[i][0] = "A"+str(i)
        labels_chunks[i][1] = "B"+str(i)        
        labels_chunks[i][2] = "\u03C4" + str(i)
        
        #Add shifted data to merged lightcurve        
        for j in range(len(mjd)):
            merged_mjd.append(mjd[j]-init_tau[i])
            merged_flux.append(flux[j])
            merged_err.append(err[j])
                
    merged_mjd = np.array(merged_mjd)
    merged_flux = np.array(merged_flux)
    merged_err = np.array(merged_err)
    
    P_slow=np.empty(len(data))
    #Calculate no. of parameters for a grid of deltas over the prior range
    if (calc_P == True):
        print("Calculating No. of parameters beforehand...")
        deltas=np.arange(priors[3][0], priors[3][1], 0.02)
        ps=np.empty(len(deltas))
        for i in tqdm(range(len(deltas))):
            ps[i]=CalculateP(merged_mjd, merged_flux, merged_err, deltas[i])
            
        #P as a func of delta
        P_func=interpolate.interp1d(deltas, ps, kind="linear", fill_value="extrapolate")
    else:
        P_func=None
        
    slow_comps =[]                    
    if (include_slow_comp==True):
        for i in range(len(data)):
            t_sl, m_sl, errs_sl = RunningOptimalAverage(data[i][:,0], data[i][:,1], data[i][:,2], slow_comp_delta)
            m_sl = m_sl - np.mean(m_sl)            
            slow_comps.append([t_sl, m_sl, errs_sl])
            P_slow[i] = CalculateP(data[i][:,0], data[i][:,1], data[i][:,2], slow_comp_delta)
            

   
                        

    pos_chunks[-1][0] = init_delta#Initial delta
    labels_chunks[-1][0] = "\u0394"
    

    pos = list(chain.from_iterable(pos_chunks))#Flatten into single array
    labels = list(chain.from_iterable(labels_chunks))#Flatten into single array
    
    pos_rem = pos_ref*Nchunk + 2
    pos = np.delete(pos, pos_rem) 
    labels = np.delete(labels, pos_rem)
    
    if (delay_dist==True):
        if (pos_ref == 0):
            pos = np.delete(pos, [2]) 
            labels = np.delete(labels, [2])
        else:        
            pos = np.delete(pos, [3]) 
            labels = np.delete(labels, [3])


    print("Initial Parameter Values")
    table = [pos]
    print(tabulate(table, headers=labels))
    
   
    

    #Define starting position
    
    pos = 1e-4 * np.random.randn(int(2.0*Npar), int(Npar - param_delete)) + pos
    nwalkers, ndim = pos.shape
    print("NWalkers="+str(int(2.0*Npar)))
    
    

    with Pool() as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[data, priors, add_var, size,sig_level, include_slow_comp, slow_comp_delta, P_func, slow_comps, P_slow, init_delta, delay_dist, pos_ref], pool=pool)
        sampler.run_mcmc(pos, Nsamples, progress=True);

    #Extract samples with burn-in of 1000
    samples_flat = sampler.get_chain(discard=Nburnin, thin=15, flat=True)

         
    samples = sampler.get_chain()
    
    
    
    #####################################################################################
    # Repeat data shifting and ROA fit using best fit parameters
    
    transpose_samples = np.transpose(samples_flat)

    if (delay_dist==True):
        if (pos_ref == 0):
            transpose_samples=np.insert(transpose_samples, [2], np.array([0.0]*len(transpose_samples[1])), axis=0)              #Insert zero for reference delay dist
        else:
            transpose_samples=np.insert(transpose_samples, [3], np.array([0.0]*len(transpose_samples[1])), axis=0)              #Insert zero for reference delay dist  

    transpose_samples= np.insert(transpose_samples, pos_rem, np.array([0.0]*len(transpose_samples[1])), axis=0)    #Insert zero for reference delay          
                     
    #Split samples into chunks
    samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)] 
    
    
    
    #Extract delta and extra variance parameters as last in params list
    if (delay_dist == True):
        delta = np.percentile(samples_chunks[-1][0], [16, 50, 84])[1]
        rmss = np.zeros(size)
        taus = np.zeros(size)
    
    else:
        delta = np.percentile(samples_chunks[-1][0], [16, 50, 84])[1]

    #Loop through each lightcurve and shift data by parameters
    merged_mjd = np.zeros(size)
    merged_flux = np.zeros(size)
    merged_err = np.zeros(size)

    params=[]
    avgs = []    
    slow_comps_out = []
    prev=0
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        tau = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]
        
        if (add_var == True):
            V =  np.percentile(samples_chunks[i][-1], [16, 50, 84])[1]
        if (delay_dist == True):
            if (i>0):
                tau_rms = np.percentile(samples_chunks[i][3], [16, 50, 84])[1]
            else:
                tau_rms=delta/100.0
                

            

        params.append([A, B, tau])    
        
        if (delay_dist == True):
            if (i>0):
                params.append([tau_rms])
            else:
                params.append([0.0])            
                    
            
        #Add extra variance
        if (add_var ==True):
            err = np.sqrt((err**2) + (V**2))
            params.append([V])


                


        if (include_slow_comp==True):
            t_slow, m_slow, errs_slow = slow_comps[i]
            m_s = interpolate.interp1d(t_slow, m_slow, kind="linear", fill_value="extrapolate")
            slow_comps_out.append(m_s)
            flux = (flux - B - m_s(mjd))/A
            
        else:
            flux = (flux - B )/A          
        #Shift data
        mjd = mjd - tau
        err = err/A


        
        for j in range(len(mjd)):
            merged_mjd[int(j+ prev)] = mjd[j]
            merged_flux[int(j+ prev)] = flux[j]
            merged_err[int(j+ prev)] = err[j]
            if (delay_dist == True):
                rmss[int(j+ prev)] = tau_rms
                taus[int(j+ prev)] = tau                
                #factors[int(j+ prev)] = delta/np.sqrt(delta**2 + (tau_rms)**2)#/delta
     
        prev = int(prev + len(mjd))
    

    params.append([delta])
            
    params = list(chain.from_iterable(params))#Flatten into single array
    
    
      
    params=np.delete(params, pos_rem)   
    
    if (delay_dist==True):
        if (pos_ref==0):
            params=np.delete(params, [2])
        else:      
            params=np.delete(params, [3])   
    #Calculate ROA to merged lc

    if (delay_dist == False):
        t, m, errs = RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)

    
    
    
        #Normalise lightcurve
        m_mean = np.mean(m)#np.average(m, weights = 1.0/(errs**2))
        m_rms = np.std(m)


        m = (m-m_mean)/m_rms
        errs = errs/m_rms
        
    else:
       # ws = CalcWind(merged_mjd, delta, rmss)
        factors, conv, x, d= CalcWinds(merged_mjd, merged_flux, merged_err, delta, rmss, len(data), sizes,  taus)       
        t,m_all,errs_all, P_all = RunningOptimalAverageConv(merged_mjd, merged_flux, merged_err, d, factors, conv, x)     

        #t,m_all,errs_all = RunningOptimalAverage3(merged_mjd, merged_flux, merged_err, delta, rmss, ws)
        #Calculate Norm. conditions 

       # m_mean = np.mean(m)
       # m_rms = np.std(m)
       # m = (m-m_mean)/m_rms
        #errs = errs/m_rms

        
        
    #Output model for specific lightcurves    
    models=[]
    prev=0
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        tau = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]       
        

        if (delay_dist == False):
            t_shifted = t + tau
            #interp = interpolate.interp1d(t_shifted, m, kind="linear", fill_value="extrapolate")
            m_m = m#interp(t)
                    
            if (include_slow_comp==True):
                m_s = np.interp(t_shifted, slow_comps[i][0], slow_comps[i][1])
                m_scaled = A*(m_m) + B + m_s
            else:
                m_scaled = A*(m_m) + B

         
            #Model
           # interp = interpolate.interp1d(t, m_scaled, kind="linear", fill_value="extrapolate")
            #model = interp(t)
            #model errors
            model_errs = errs*A
            models.append([t_shifted, m_scaled, model_errs])
            
        
            
            
            
        else:
            if (i>0):
                tau_rms = np.percentile(samples_chunks[i][3], [16, 50, 84])[1]
            else:
                tau_rms=delta/100.0
            delta_new = np.sqrt(delta**2 + (tau_rms)**2)

            
            mx=max(merged_mjd)
            mn=min(merged_mjd)
            length = abs(mx-mn)
            t = np.arange(mn, mx, length/(1000)) 

            
            ts, Xs, errss = RunningOptimalAverageOutConv(t, merged_mjd, merged_flux, merged_err, factors, conv, prev, x, delta_new)
            
            
            
            m_mean = np.mean(m_all[prev : int(prev + len(mjd))])
            m_rms = np.std(m_all[prev : int(prev + len(mjd))])


            Xs = (Xs-m_mean)/m_rms
            errss = errss/m_rms
            
            if (include_slow_comp==True):
                t_shifted = t + tau
                m_s = np.interp(t_shifted, slow_comps[i][0], slow_comps[i][1])
                
                model = A*Xs + B + m_s
            else:
                model = A*Xs + B
        
            model_errs = errss*A
            models.append([t+tau, model, model_errs])         
            
            if (i ==0):         
                t,m,errs = [t+tau, Xs,errss]
                       
        prev = int(prev + len(mjd))
        
        
    print("Best Fit Parameters")
    table = [params]
    print(tabulate(table, headers=labels))
    
    #Write samples to file
    filehandler = open(b"samples_flat.obj","wb")
    pickle.dump(samples_flat,filehandler)
    filehandler = open(b"samples.obj","wb")
    pickle.dump(samples,filehandler)

    filehandler = open(b"X_t.obj","wb")
    pickle.dump([t, m, errs],filehandler)
    if (include_slow_comp==True):      
        filehandler = open(b"Slow_Comps.obj","wb")
        pickle.dump(slow_comps_out,filehandler)
        
    filehandler = open(b"Lightcurve_models.obj","wb")
    pickle.dump(models,filehandler)
    
    
    #Plot Corner Plot
    plt.rcParams.update({'font.size': 15})
    #Save Cornerplot to figure
    fig = corner.corner(samples_flat, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 20});
    i = 1
    while os.path.exists('{}{:d}.pdf'.format("CornerPlot", i)):
        i += 1
    fig.savefig('{}{:d}.pdf'.format("CornerPlot", i))
    plt.close();
    
    
    #print("Autocorrelation time: ", sampler.get_autocorr_time())
    return samples, samples_flat, t, m, errs, slow_comps, params, models






class Fit():
    def __init__(self, datadir, objName, filters, priors, init_tau = None, init_delta=1.0, delay_dist=False , add_var=True, sig_level = 4.0, Nsamples=10000, Nburnin=5000, include_slow_comp=False, slow_comp_delta=30.0, delay_ref = None, calc_P=False):
        self.datadir=datadir
        self.objName=objName
        self.filters=filters
        data=[]
        for i in range(len(filters)):
            file = datadir + str(self.objName) +"_"+ str(self.filters[i]) + ".dat"
            data.append(np.loadtxt(file))

           # np.savetxt(datadir + str(self.objName) +"_"+ str(self.filters[i]) + ".dat",np.transpose([data[i][:,0], data[i][:,1], data[i][:,2]]))
            
            
        
        self.priors= priors
        self.init_tau = init_tau
        self.init_delta=init_delta
        
       # if (add_var == True):
          #  self.add_var = [True]*len(filters)
       # elif (add_var == False):
         #   self.add_var = [False]*len(filters)
     #   else:
        self.add_var = add_var
            
            
        self.delay_dist = delay_dist

        
        
        self.sig_level = sig_level
        self.Nsamples = Nsamples
        self.Nburnin = Nburnin
        
        if (delay_ref == None):
            self.delay_ref = filters[0]
        else:
            self.delay_ref = delay_ref
        self.delay_ref_pos = np.where(np.array(filters) == self.delay_ref)[0]
        if (init_tau == None):
            self.init_tau = [0]*len(data)
        else:
            Nchunk = 3
            if (self.add_var == True):
                Nchunk +=1
            if (self.delay_dist == True):
                Nchunk+=1
        
            self.init_tau = np.insert(init_tau, self.delay_ref_pos, 0.0)
            
        self.include_slow_comp=include_slow_comp
        self.slow_comp_delta=slow_comp_delta
        
        self.calc_P=calc_P
        
        
        run = FullFit(data, self.priors, self.init_tau, self.init_delta, self.add_var, self.sig_level, self.Nsamples, self.Nburnin, self.include_slow_comp, self.slow_comp_delta, self.calc_P, self.delay_dist, self.delay_ref_pos)

        self.samples = run[0]
        self.samples_flat = run[1]
        self.t = run[2]
        self.X = run[3]
        self.X_errs = run[4]
        if (self.include_slow_comp==True):
            self.slow_comps=run[5]
        self.params=run[6]
        self.models = run[7]
        
        
        
        
#Plotting


def Plot(Fit):

    plt.style.use(['science','no-latex'])        
    plt.rcParams.update({
            "font.family": "Sans", 
            "font.serif": ["DejaVu"],
            "figure.figsize":[15,20],
            "font.size": 20,
            "xtick.major.size" : 6,
            "xtick.major.width": 1.2,
            "xtick.minor.size" : 3,
            "xtick.minor.width" : 1.2,
            "ytick.major.size" : 6,
            "ytick.major.width": 1.2,
            "ytick.minor.size" : 3,
            "ytick.minor.width" : 1.2}) 






    datadir =Fit.datadir
    objName = Fit.objName
    filters=Fit.filters
    data=[]
    for i in range(len(filters)):
        file = datadir + str(objName) +"_"+ str(filters[i]) + ".dat"
        data.append(np.loadtxt(file))


    cmap = matplotlib.cm.get_cmap('tab10')
    band_colors=[]
    n = np.arange(0.05, 1.0 + 0.5/len(filters), 1.0/len(filters))
    for i in range(len(filters)):
        band_colors.append(cmap(n[i]))

    samples_flat = Fit.samples_flat
    t = Fit.t
    X=Fit.X
    errs= Fit.X_errs
    
    
    transpose_samples = np.transpose(samples_flat)      
    
        
    Nchunk = 3
    if (Fit.add_var == True):
        Nchunk +=1
    if (Fit.delay_dist == True):
        Nchunk+=1
        
        if (Fit.delay_ref_pos == 0):
            transpose_samples=np.insert(transpose_samples, [2], np.array([0.0]*len(transpose_samples[1])), axis=0)              #Insert zero for reference delay dist
        else:
            transpose_samples=np.insert(transpose_samples, [3], np.array([0.0]*len(transpose_samples[1])), axis=0)              #Insert zero for reference delay dist
        
        
        param_delete=2
    else:
        param_delete=1
        


                 
    Npar =  Nchunk*len(data) + 1
    pos = Fit.delay_ref_pos*Nchunk + 2
    transpose_samples= np.insert(transpose_samples, pos, np.array([0.0]*len(transpose_samples[1])), axis=0)    #Insert zero for reference delay 
    
    

        
    chunk_size = int((Npar - 1)/len(data))
        

        
           
    samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)]        
        
    fig = plt.figure(100)
    gs = fig.add_gridspec(len(filters), 2, hspace=0, wspace=0, width_ratios=[5, 1])
    axs= gs.subplots(sharex='col') 
    tss=[]    
    for j in range(len(filters)):

        #Read in parameter values
        A = np.percentile(samples_chunks[j][0], [16, 50, 84])[1]
        B = np.percentile(samples_chunks[j][1], [16, 50, 84])[1]
        tau = np.percentile(samples_chunks[j][2], [16, 50, 84])       
        tss.append(tau[1])
        tau_samples=samples_chunks[j][2]
        
        mjd = data[j][:,0]
        flux = data[j][:,1]
        err = data[j][:,2] 
        
        
        #Add extra variance
        if (Fit.add_var == True):
            sig = np.percentile(samples_chunks[j][-1], [16, 50, 84])[1]  
            err = np.sqrt(err**2 + sig**2)
        
        ts, model, errs = Fit.models[j]
        

        axs[j][0].errorbar(mjd, flux , yerr=err, ls='none', marker=".", color=band_colors[j])
        axs[j][0].plot(ts, model, color="black")
        axs[j][0].fill_between(ts , model+errs,  model-errs, facecolor="darkgrey", edgecolor='none', rasterized=True, antialiased=True)
        
        if (Fit.include_slow_comp == True):
            slow_comp = Fit.slow_comps[j]
            axs[j][0].plot(t, slow_comp(t)+B, linestyle="dashed", color="black")          
        
        length=abs(max(flux)-min(flux))
        axs[j][0].set_ylim(min(flux)-0.2*length, max(flux)+0.2*length)
        axs[j][0].set_xlabel("MJD")
        
        axs[j][0].annotate(filters[j], xy=(0.85, 0.85), xycoords='axes fraction', size=15.0, color=band_colors[j], fontsize=20) 
        
        frq, edges = np.histogram(tau_samples, bins=50)        
        if (Fit.delay_dist==True):
            tau_rms = np.percentile(samples_chunks[j][3], [16, 50, 84])
            norm = 1.0 / ((tau_rms[1]) * np.sqrt(2.0 * np.pi))
            norm = norm/max(frq)
            if (Fit.delay_ref_pos>0 and j==0):
                norm = 1.0
            
        else:
            norm=1.0/max(frq)

        
                
        

        axs[j][1].bar(edges[:-1], frq*norm, width=np.diff(edges), edgecolor=band_colors[j], align="edge", color=band_colors[j])  
        #axs[j][1].hist(tau_samples, color=band_colors[j], bins=50, density=True)
        axs[j][1].axvline(x = np.percentile(tau_samples, [16, 50, 84])[1], color="black")
        axs[j][1].axvline(x = np.percentile(tau_samples, [16, 50, 84])[0] , color="black", ls="--")
        axs[j][1].axvline(x = np.percentile(tau_samples, [16, 50, 84])[2], color="black",ls="--")
        axs[j][1].axvline(x = 0, color="black",ls="--")    
        axs[j][1].set_xlabel("Time Delay (Days)")

        if (Fit.delay_dist==True):
            if (j>0):

                tau_rms = np.percentile(samples_chunks[j][3], [16, 50, 84])
            
                length=10.0*tau_rms[1]
                taus=np.arange(tau[1] - 5.0*tau_rms[1], tau[1] + 5.0*tau_rms[1], length/500)
                
                
                #G=models.Gaussian1D(amplitude=1 / ((tau_rms[1]) * np.sqrt(2 * np.pi)), mean=tau[1], stddev=tau_rms[1])

            
                #Limits for errors
                up=[]
                low=[]
                Gs = []
                for k in range(len(taus)):
                    rms_samples = samples_chunks[j][3]
                    mean_samples = samples_chunks[j][2]
                    cutoff_samples = samples_chunks[0][2]



                    if (taus[k]>=tss[0]):
                        G=1.0/((rms_samples) * np.sqrt(2 * np.pi)) *np.exp(-0.5*((taus[k] - mean_samples)/rms_samples)**2)
                    else:
                        G=np.zeros(len(mean_samples))
                    percent =  np.percentile(G, [16, 50, 84])
                    up.append(percent[0])
                    Gs.append(percent[1])
                    low.append(percent[2])
                    
                    
                    
                axs[j][1].plot( taus, Gs, color="black", lw=1.5)                    
                
                axs[j][1].fill_between(taus, up, low, color="black", alpha=0.5, edgecolor='none', rasterized=True, antialiased=True)
    length = max(tss)-min(tss)        
    axs[-1][1].set_xlim(min(tss)-0.5*length, max(tss)+0.5*length)
            
            
            
            
            
            
            
        
    for ax in axs.flat:
        ax.label_outer()    
      
    plt.tight_layout() 
    plt.show() 
        
##################################################################################################################################
##################################################################################################################################



############################################### Intercalibration Model ####################################################
      
        
        
#Log Likelihood
def log_likelihood2(params, data, sig_level):

    #Break params list into chunks of 3 i.e A, B, V in each chunk
    params_chunks = [params[i:i + 3] for i in range(0, len(params), 3)] 
    
    #Extract delta parameter as last in params list
    delta = params_chunks[-1][0]
    

    #Loop through each lightcurve and shift data by parameters
    merged_mjd = []
    merged_flux = []
    merged_err = []
    avgs=[]
    for i in range(len(data)):
        A = params_chunks[i][0]
        B = params_chunks[i][1]
        sig = params_chunks[i][2] 
        
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]

        #Shift data
        flux = (flux - B)/A
        #Add extra variance
        err = np.sqrt((err**2) + (sig**2))
        
        err = err/A        
       
        
                
        #Add shifted data to merged lightcurve
        for j in range(len(mjd)):
            merged_mjd.append(mjd[j])
            merged_flux.append(flux[j])
            merged_err.append(err[j])        
    
    merged_mjd = np.array(merged_mjd)
    merged_flux = np.array(merged_flux)
    merged_err = np.array(merged_err)


    
    #Calculate ROA to merged lc
    t, m, errs = RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
    P=CalculateP(merged_mjd, merged_flux, merged_err, delta)

    
    
    

    #Calculate chi-squared for each lightcurve and sum
    lps=[0]*len(data)
    for i in range(len(data)):

        A = params_chunks[i][0]
        B = params_chunks[i][1] 
        sig = params_chunks[i][2] 

        #Originial lightcurves
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        #Add extra variance
        err = np.sqrt((err**2) + (sig**2))
        
        #Scale and shift model        
        m_scaled = A*(m) + B
        
         
              #Model
        interp = interpolate.interp1d(t, m_scaled, kind="linear", fill_value="extrapolate")
        model = interp(mjd)
        chi2 = np.empty(len(mjd))
        ex_term = np.empty(len(mjd))  
        for j in range(len(mjd)):

            if(abs(model[j]-flux[j]) < sig_level*err[j]):
            
            
                chi2[j] = ((model[j]-flux[j])**2)/(err[j]**2)
                
                ex_term[j] = np.log(2.0*np.pi*(err[j]**2))  
                              
            else:
                chi2[j] =sig_level**2
                ex_term[j] = np.log(2.0*np.pi*((abs(model[j] - flux[j])/sig_level)**2))
        lps[i]=np.sum(chi2 + ex_term) 
    
    lprob = np.sum(lps)  
    
    #Calculate Penalty
    Penalty = 0.0
    for i in range(len(data)):
        mjd = data[i][:,0]

        Penalty = Penalty + 3.0*np.log(len(mjd))
            
    Penalty = Penalty + (P*np.log(len(merged_flux)))
        
    BIC =  lprob + Penalty

    return -1.0*BIC
    
    
 
#Priors
def log_prior2(params, priors, s):
    #Break params list into chunks of 3 i.e A, B, tau in each chunk
    params_chunks = [params[i:i + 3] for i in range(0, len(params), 3)]
    
    #Extract delta and extra variance parameters as last in params list
    delta = params_chunks[-1][0]

        
    #Read in priors
    sig_prior = priors[1]
    delta_prior = priors[0]
    A=[]
    B=[]
    V0=[]
    
    check=[]
    A_prior = []
    B_prior=[]
    #Loop over lightcurves
    for i in range(s):
        A = params_chunks[i][0]
        B = params_chunks[i][1]
        sig = params_chunks[i][2]
        
        B_prior_width=0.5 # mJy
        lnA_prior_width=0.02 # 0.02 = 2%
        
        A_prior.append(-2.0*np.log(lnA_prior_width*A*np.sqrt(2.0*np.pi)) - (np.log(A)/lnA_prior_width)**2.0)
        B_prior.append(2.0*np.log((1.0/np.sqrt(2.0*np.pi*(B_prior_width**2)))*np.exp(-0.5*(B/B_prior_width)**2)))
        
        if sig_prior[0] < sig < sig_prior[1]:
            check.append(0.0)
        else:
            check.append(1.0)
            
    A_prior = np.array(A_prior)
    B_prior = np.array(B_prior)    
          
    if np.sum(np.array(check)) == 0.0 and delta_prior[0]< delta < delta_prior[1]:
        return np.sum(A_prior) + np.sum(B_prior)
    else:
        return -np.inf

    

    
    
#Probability
def log_probability2(params, data, priors, sig_level):
    lp = log_prior2(params, priors, len(data))
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood2(params, data, sig_level)
    
    
    
        



def InterCalib(data, priors, init_delta, sig_level, Nsamples, Nburnin, filter):

    ########################################################################################    
    #Run MCMC to fit to data
    Npar = 3*len(data) + 1
    
    #Set inital conditions
    pos = [0]*(3*len(data) + 1)
    labels = [None]*(3*len(data) + 1)
    pos_chunks = [pos[i:i + 3] for i in range(0, len(pos), 3)]
    labels_chunks = [labels[i:i + 3] for i in range(0, len(labels), 3)]
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
                
        pos_chunks[i][0] = pos_chunks[i][0] + 1.0 #Set intial A to one
        pos_chunks[i][1] = pos_chunks[i][1] + 0.0 #Set initial B to zero  
        pos_chunks[i][2] = pos_chunks[i][1] + 0.01#2 #Set initial V to 0.02   
        
        labels_chunks[i][0] = "A"+str(i+1)
        labels_chunks[i][1] = "B"+str(i+1)        
        labels_chunks[i][2] = "\u03C3"+str(i+1)                
        
    pos_chunks[-1][0] = init_delta#Initial delta
    labels_chunks[-1][0] = "\u0394"
    pos = list(chain.from_iterable(pos_chunks))#Flatten into single array
    labels = list(chain.from_iterable(labels_chunks))#Flatten into single array     
    

    print("Initial Parameter Values")
    table = [pos]
    print(tabulate(table, headers=labels))

    #Define starting position
    pos = 1e-4 * np.random.randn(int(2.0*Npar), int(Npar)) + pos
    print("NWalkers="+str(int(2.0*Npar)))
    nwalkers, ndim = pos.shape
    with Pool() as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability2, args=(data, priors, sig_level), pool=pool)
        sampler.run_mcmc(pos, Nsamples, progress=True);
    
    #Extract samples with burn-in of 1000
    samples_flat = sampler.get_chain(discard=Nburnin, thin=15, flat=True)
            
    samples = sampler.get_chain()
    
    
    
    #####################################################################################
    # Repeat data shifting and ROA fit using best fit parameters
    
    
    #Split samples into chunks
    samples_chunks = [np.transpose(samples_flat)[i:i + 3] for i in range(0, len(np.transpose(samples_flat)), 3)] 
    merged_mjd = []
    merged_flux = []
    merged_err = []
    A_values = []
    B_values = []
    avgs = []
    params = []
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        A_values.append(A)
        B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        B_values.append(B)
        sig = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]
        params.append([A, B, sig])
        #Shift data
        flux = (flux - B)/A
        #Add extra variance
        err = np.sqrt((err**2) + (sig**2))
        err = err/A
        
        avgs.append(np.average(flux, weights = 1.0/(err**2)))
        #Add shifted data to merged lightcurve        
        for j in range(len(mjd)):
            merged_mjd.append(mjd[j])
            merged_flux.append(flux[j])
            merged_err.append(err[j])
                
    merged_mjd = np.array(merged_mjd)
    merged_flux = np.array(merged_flux)
    merged_err = np.array(merged_err)
    A_values = np.array(A_values)
    B_values = np.array(B_values)
       
    delta = np.percentile(samples_chunks[-1], [16, 50, 84])[1]
    params.append([delta])
    params = list(chain.from_iterable(params))#Flatten into single array
    #Calculate ROA to merged lc
    t, m, errs = RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
    
    Calibrated_mjd = []
    Calibrated_flux = []
    Calibrated_err = [] 
    
      

    
    for i in range(len(data)):
        A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        sig = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]
        
        
        #Originial lightcurves
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        #Add extra variance
        err = np.sqrt((err**2) + (sig**2))
     
        m_scaled = A*(m) + B

        #Model
        interp = interpolate.interp1d(t, m_scaled, kind="linear", fill_value="extrapolate")
        model = interp(mjd)
        
        #Sigma Clipping
        mask = (abs(model - flux) < sig_level*err)

        
        #Shift by parameters
        flux = (flux - B)/A          

        no_clipped = 0.0
        for j in range(len(mask)):
            if (mask[j]==False):
                no_clipped = no_clipped + 1
        print(no_clipped, "clipped, out of ", len(mjd), "data points")
        
        #Add shifted data to merged lightcurve        
        for j in range(len(mjd)):
            Calibrated_mjd.append(mjd[j])
            Calibrated_flux.append(flux[j])
            if (abs(model[j] - flux[j]) > sig_level*err[j]):
                Calibrated_err.append((abs(model[j] - flux[j])/sig_level))
            else:
                Calibrated_err.append(err[j])
                
    Calibrated_mjd = np.array(Calibrated_mjd)
    Calibrated_flux = np.array(Calibrated_flux)
    Calibrated_err = np.array(Calibrated_err)
                
                    
    print("<A> = ", np.mean(A_values))
    print("<B> = ", np.mean(B_values))
    

    plt.rcParams.update({'font.size': 15})
    #Save Cornerplot to figure
    fig = corner.corner(samples_flat, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 20}, truths=params);
    i = 0
    while os.path.exists('{}{:d}.pdf'.format(str(filter)+"_Calibration_CornerPlot", i)):
        i += 1
    fig.savefig('{}{:d}.pdf'.format(str(filter)+"_Calibration_CornerPlot", i))
    plt.close();
           

    return samples, samples_flat, t, m, errs, Calibrated_mjd, Calibrated_flux, Calibrated_err





class InterCalibrate():
    def __init__(self, datadir, objName, filter, scopes, priors, init_delta=1.0, sig_level = 4.0, Nsamples=15000, Nburnin=10000):
        self.datadir=datadir
        self.objName=objName
        self.filter=filter
        self.scopes=scopes
        data=[]
        for i in range(len(scopes)):
            file = datadir + str(self.objName) +"_"+ str(self.filter) + "_"+ str(self.scopes[i]) +".dat"
            #Check if file is empty
            if os.stat(file).st_size == 0:
                print("")
            else:
                data.append(np.loadtxt(file))
            

        
        self.priors= priors
        self.init_delta=init_delta
        self.sig_level = sig_level
        self.Nsamples = Nsamples
        self.Nburnin = Nburnin

        
        run = InterCalib(data, self.priors, self.init_delta, self.sig_level, self.Nsamples, self.Nburnin, self.filter)

        self.samples = run[0]
        self.samples_flat = run[1]
        self.t = run[2]
        self.X = run[3]
        self.X_errs = run[4]
        self.mjd=run[5]
        self.flux=run[6]
        self.err=run[7]
        
        #Write to file
        np.savetxt(datadir + str(self.objName) +"_"+ str(self.filter) +  ".dat", np.transpose([run[5], run[6], run[7]]))
        
        plt.rcParams.update({
            "font.family": "Sans", 
            "font.serif": ["DejaVu"],
            "figure.figsize":[20,10],
            "font.size": 20})          
        
        #Plot calibrated ontop of original lcs
        fig=plt.figure(2)
        plt.title(str(filter))
        #Plot data for filter
        for i in range(len(data)):
            mjd = data[i][:,0]
            flux = data[i][:,1]
            err = data[i][:,2]
            plt.errorbar(mjd, flux, yerr=err, ls='none', marker=".", label=str(scopes[i]), alpha=0.5)

        plt.errorbar(run[5], run[6], yerr=run[7], ls='none', marker=".", color="black", label="Calibrated")

        plt.xlabel("mjd")
        plt.ylabel("Flux")
        plt.legend()
        
        
        i = 0
        while os.path.exists('{}{:d}.pdf'.format(str(filter)+"_Calibration_Plot", i)):
            i += 1
        fig.savefig('{}{:d}.pdf'.format(str(filter)+"_Calibration_Plot", i))
        plt.close();     
        

        



























##################################################################################################################################
##################################################################################################################################



############################################### Grav. Lensing Model ##############################################################
      



#Log Likelihood
def log_likelihood3(params, data, add_var, size, sig_level):


    if (add_var == True):
        Npar = 7*len(data) + 3
    else:
        Npar = 6*len(data) + 3
        
    chunk_size = int((Npar - 3)/len(data))
    
       

    #Break params list into chunks of 3 i.e A, B, tau in each chunk
    params_chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size )] 
    
    #Extract delta and extra variance parameters as last in params list
    delta = params_chunks[-1][2]
    A1 = params_chunks[-1][0]
    B1 = params_chunks[-1][1]

    #Loop through each lightcurve and shift data by parameters
    merged_mjd = np.zeros(size)
    merged_flux = np.zeros(size)
    merged_err = np.zeros(size)
    prev=0
    for i in range(len(data)):
        tau = params_chunks[i][0]
        if (add_var == True):
            V =  params_chunks[i][-1]

        #Extract polynomial coefficients
        

        P0 = params_chunks[i][1]
        P1 = params_chunks[i][2]
        P2 = params_chunks[i][3]
        P3 = params_chunks[i][4]
        P4 = params_chunks[i][5] 
               
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]

        #Calculate normalised time
        eta = (2.0*mjd -(max(mjd) + min(mjd)))/(max(mjd) - min(mjd))
        
        #Calculate Polynomials
      

        poly=P0 + P1*(eta**1) + P2*(eta**2)+ P3*(eta**3) + P4*(eta**4)

        #poly_1 = params_chunks[0][1] + params_chunks[0][2]*(eta**1) + params_chunks[0][3]*(eta**2)+ params_chunks[0][4]*(eta**3) + params_chunks[0][5]*(eta**4)

        #Add extra variance
        if (add_var == True):

            err = np.sqrt((err**2) + (V**2))
            
                    
        #tau = tau - params_chunks[0][0]
        #Shift data

        mjd = mjd - tau
        flux = flux/(10**(-0.4*(poly)))
        err = err/(10**(-0.4*(poly)))         
        flux = (flux - B1)/(A1) 
        err = err/A1

   
        
        #Add shifted data to merged lightcurve
        for j in range(len(mjd)):
            merged_mjd[int(j+ prev)] = mjd[j]
            merged_flux[int(j+ prev)] = flux[j]
            merged_err[int(j+ prev)] = err[j]
     
        prev = int(prev + len(mjd))
        


    
    #Calculate ROA to merged lc
    t, m, errs = RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
    P=CalculateP(merged_mjd, merged_flux, merged_err, delta)


    #Normalise lightcurve

    m_mean = np.mean(m)#np.average(m, weights = 1.0/(errs**2))
    m_rms = np.std(m)
    m = (m-m_mean)/m_rms
    errs = errs/m_rms


    #Calculate chi-squared for each lightcurve and sum
    lps=[0]*len(data)
    for i in range(len(data)):
        tau = params_chunks[i][0]
        if (add_var == True):
            V =  params_chunks[i][-1]

        #Extract polynomial coefficients

        P0 = params_chunks[i][1]
        P1 = params_chunks[i][2]
        P2 = params_chunks[i][3]
        P3 = params_chunks[i][4]
        P4 = params_chunks[i][5] 

        #Originial lightcurves
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        #Calculate normalised time
        eta = (2.0*t -(max(mjd) + min(mjd)))/(max(mjd) - min(mjd))
                
        #Add extra variance
        if (add_var == True):
            err = np.sqrt((err**2) + (V**2))  
                  
          

        t_shifted = t + tau
        interp = interpolate.interp1d(t_shifted, m, kind="linear", fill_value="extrapolate")
        m_m = interp(t)        
                 

        poly=P0 + P1*(eta**1) + P2*(eta**2)+ P3*(eta**3) + P4*(eta**4)       

        m_scaled = (10**(-0.4*(poly)))*(A1*m_m + B1)
        t_shifted = t        


         
        #Model
        interp = interpolate.interp1d(t_shifted, m_scaled, kind="linear", fill_value="extrapolate")
        model = interp(mjd)

        chi2 = np.empty(len(mjd))
        ex_term = np.empty(len(mjd))  
        for j in range(len(mjd)):

            if(abs(model[j]-flux[j]) < sig_level*err[j]):
            
            
                chi2[j] = ((model[j]-flux[j])**2)/(err[j]**2)
                
                ex_term[j] = np.log(((err[j]**2)/(data[i][j,2]**2)))  
                              
            else:
                chi2[j] =sig_level**2
                ex_term[j] = np.log(((abs(model[j] - flux[j])/sig_level)**2)/(data[i][j,2]**2))
        lps[i]=np.sum(chi2 + ex_term) 
    
    lprob = np.sum(lps) 



    Penalty = 0.0
    for i in range(len(data)):
        mjd = data[i][:,0]
        
        if (i==0):
            Penalty = Penalty + (chunk_size-5.0)*np.log(len(mjd))

        else:
            Penalty = Penalty + chunk_size*np.log(len(mjd))
            
    Penalty = Penalty + ((P+2.0)*np.log(len(merged_flux)))
        
    BIC =  lprob + Penalty

    return -1.0*BIC
    
    
 
#Priors
def log_prior3(params, priors, add_var, data):

    if (add_var == True):
        Npar = 7*len(data) + 3
    else:
        Npar = 6*len(data) + 3
        
    chunk_size = int((Npar - 3)/len(data))
    

    #Break params list into chunks of 3 i.e A, B, tau in each chunk
    params_chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size)]

    #Extract delta and extra variance parameters as last in params list
    delta = params_chunks[-1][2]
    A0 = params_chunks[-1][0]
    B0 = params_chunks[-1][1]
        
    #Read in priors
    A_prior = priors[0]
    B_prior = priors[1]
    tau_prior = priors[2]
    delta_prior = priors[3]
    V_prior = priors[4]
    P_prior = priors[5]
    

    check=[]
    #Loop over lightcurves
    for i in range(len(data)):
        tau = params_chunks[i][0]
        if (add_var == True):
            V =  params_chunks[i][-1]
        #Extract polynomial coefficients
        P0 = params_chunks[i][1]
        P1 = params_chunks[i][2]
        P2 = params_chunks[i][3]
        P3 = params_chunks[i][4]
        P4 = params_chunks[i][5]
        

                     
        if (add_var == True):
            if tau_prior[0] <= tau <= tau_prior[1] and V_prior[0]<= V <= V_prior[1] and P_prior[0] < P0 <= P_prior[1]and P_prior[0] < P1 <= P_prior[1]and P_prior[0] < P2 <= P_prior[1]and P_prior[0] < P3 <= P_prior[1]and P_prior[0] < P4 <= P_prior[1]:
                check.append(0.0)
            else:
                check.append(1.0)
        else:
            if tau_prior[0] <= tau <= tau_prior[1] and P_prior[0] < P0 <= P_prior[1]and P_prior[0] < P1 <= P_prior[1]and P_prior[0] < P2 <= P_prior[1]and P_prior[0] < P3 <= P_prior[1]and P_prior[0] < P4 <= P_prior[1]:
                check.append(0.0)
            else:
                check.append(1.0)        
            

    if np.sum(np.array(check)) == 0.0 and delta_prior[0]<= delta <= delta_prior[1] and A_prior[0]<=A0<=A_prior[1] and B_prior[0]<=B0<=B_prior[1]:
        return 0.0
    else:
        return -np.inf

    

    
    
#Probability
def log_probability3(params, data, priors, add_var, size, sig_level):

    params=np.insert(params, [0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    lp = log_prior3(params, priors, add_var, data)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood3(params, data, add_var, size, sig_level)
    
    



     

def LensFit(data, priors, init_tau, init_delta, add_var, sig_level, Nsamples, Nburnin, image, file):

    if (add_var == True):
        Npar = 7*len(data) + 3
    else:
        Npar = 6*len(data) + 3
        
    chunk_size = int((Npar - 3)/len(data))
     

    ########################################################################################    
    #Run MCMC to fit to data
    
    #Choose intial conditions from mean and rms of data
    pos = [0]*Npar
    labels = [None]*Npar
    labels_latex = [None]*Npar
    chunk_size = int((Npar - 3)/len(data))
    
    pos_chunks = [pos[i:i + chunk_size] for i in range(0, len(pos), chunk_size)]
    labels_chunks = [labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)]
    labels_chunks_latex = [labels_latex[i:i + chunk_size] for i in range(0, len(labels_latex), chunk_size)]    
    merged_mjd = []
    merged_flux = []
    merged_err = []
    size = 0
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        size = size + len(mjd)
                
        pos_chunks[i][0] = init_tau[i]

        if(add_var == True):
            pos_chunks[i][-1] = 0.02 #Set initial V
            labels_chunks[i][-1] = "\u03C3"+str(i+1)
            labels_chunks_latex[i][-1] = "$\sigma_"+str(i+1)+"$"
            
            
        #Find initial polynomial coeff.
        
        
        t,m,errs = RunningOptimalAverage(data[0][:,0], data[0][:,1], data[0][:,2], init_delta)
        t_shifted = t + init_tau[i]
        interp = interpolate.interp1d(t_shifted, m, kind="linear", fill_value="extrapolate")
        m1 = interp(mjd)    
        
        delta_m = -2.5*np.log10(data[i][:,1]/m1)
        eta = (2.0*mjd -(max(mjd) + min(mjd)))/(max(mjd) - min(mjd))
        
        coeffs = np.polyfit(eta, delta_m, 4)

        if (i==0):

            pos_chunks[i][1] = 0.0
            pos_chunks[i][2] = 0.0
            pos_chunks[i][3] = 0.0
            pos_chunks[i][4] = 0.0
            pos_chunks[i][5] = 0.0
        else:
            pos_chunks[i][1] = coeffs[4]
            pos_chunks[i][2] = coeffs[3]
            pos_chunks[i][3] = coeffs[2]
            pos_chunks[i][4] = coeffs[1]
            pos_chunks[i][5] = coeffs[0]          


        
        labels_chunks[i][1] = "P_" + str(i) + str(0)
        labels_chunks[i][2] = "P_" + str(i) + str(1)        
        labels_chunks[i][3] = "P_" + str(i) + str(2)
        labels_chunks[i][4] = "P_" + str(i) + str(3)        
        labels_chunks[i][5] = "P_" + str(i) + str(4)       
         
        labels_chunks_latex[i][1] = "$P_{" + str(i) + str(0)+"}$"
        labels_chunks_latex[i][2] = "$P_{" + str(i) + str(1)  +"}$"      
        labels_chunks_latex[i][3] = "$P_{" + str(i) + str(2)+"}$"
        labels_chunks_latex[i][4] = "$P_{" + str(i) + str(3) +"}$"       
        labels_chunks_latex[i][5] = "$P_{" + str(i) + str(4) +"}$"                                                  
     
        labels_chunks[i][0] = "\u03C4" + str(i)                
        labels_chunks_latex[i][0] = "$\\tau_" + str(i+1) +"$"                
        #Add shifted data to merged lightcurve        
        for j in range(len(mjd)):
            merged_mjd.append(mjd[j]-init_tau[i])
            merged_flux.append(flux[j])
            merged_err.append(err[j])
                
    merged_mjd = np.array(merged_mjd)
    merged_flux = np.array(merged_flux)
    merged_err = np.array(merged_err)
    
    
    
    pos_chunks[-1][0] = np.std(data[0][:,1])#Initial A0
    labels_chunks_latex[-1][0] = "$A_1$"
    labels_chunks[-1][0] = "A1"    
    pos_chunks[-1][1] = np.mean(data[0][:,1])#Initial A0
    labels_chunks_latex[-1][1] = "$B_1$"     
    labels_chunks[-1][1] = "B1"     
    
    pos_chunks[-1][2] = init_delta#Initial delta
    labels_chunks_latex[-1][2] = "$\Delta$"
    labels_chunks[-1][2] = "\u0394"    
    #Remove tau for first lightcurve and flatten   

    pos = list(chain.from_iterable(pos_chunks))#Flatten into single array
    labels = list(chain.from_iterable(labels_chunks))#Flatten into single array
    labels_latex = list(chain.from_iterable(labels_chunks_latex))#Flatten into single array   
    pos = np.delete(pos, [ 0,1, 2, 3, 4, 5])    

    labels = np.delete(labels, [0,1, 2, 3, 4, 5])
    labels_latex = np.delete(labels_latex, [0,1, 2, 3, 4, 5])       

    print("Initial Parameter Values")
    table = [pos]
    print(tabulate(table, headers=labels))

    #Define starting position
    pos = 1e-4 * np.random.randn(int(2.0*Npar), Npar - 6) + pos
    nwalkers, ndim = pos.shape
    
    print("Nwalkers = ", nwalkers)  
    with Pool() as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability3, args=(data, priors, add_var, size, sig_level), pool=pool)
        sampler.run_mcmc(pos, Nsamples, progress=True);
    
    #Extract samples with burn-in of 1000
    samples_flat = sampler.get_chain(discard=Nburnin, thin=15, flat=True)
            
    samples = sampler.get_chain()
    
    
    
    #####################################################################################
    # Repeat data shifting and ROA fit using best fit parameters
    
    transpose_samples = np.transpose(samples_flat)      
    transpose_samples = np.insert(transpose_samples, [0], np.array([0.0]*len(transpose_samples[1])), axis=0)
    transpose_samples = np.insert(transpose_samples, [1], np.array([0.0]*len(transpose_samples[1])), axis=0)    
    transpose_samples = np.insert(transpose_samples, [2], np.array([0.0]*len(transpose_samples[1])), axis=0)    
    transpose_samples = np.insert(transpose_samples, [3], np.array([0.0]*len(transpose_samples[1])), axis=0)
    transpose_samples = np.insert(transpose_samples, [4], np.array([0.0]*len(transpose_samples[1])), axis=0)    
    transpose_samples = np.insert(transpose_samples, [5], np.array([0.0]*len(transpose_samples[1])), axis=0)      
    
    #Split samples into chunks
    samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)] 
    merged_mjd = []
    merged_flux = []
    merged_err = []

    params=[]
    avgs = []
    data_fluxes=[]
    A1 = np.percentile(samples_chunks[-1][0], [16, 50, 84])[1]   
    B1 = np.percentile(samples_chunks[-1][1], [16, 50, 84])[1]   
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        tau = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        if (add_var ==True):
            V = np.percentile(samples_chunks[i][-1], [16, 50, 84])[1]
            err = np.sqrt((err**2) + (V**2))            
        #Extract polynomial coefficients

        P0 = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        P1 = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]
        P2 = np.percentile(samples_chunks[i][3], [16, 50, 84])[1]
        P3 = np.percentile(samples_chunks[i][4], [16, 50, 84])[1]      
        P4 = np.percentile(samples_chunks[i][5], [16, 50, 84])[1]
             
        #Calculate normalised time
        eta = (2.0*mjd -(max(mjd) + min(mjd)))/(max(mjd) - min(mjd))
        
        #Shift data
        

        poly=P0 + P1*(eta**1) + P2*(eta**2)+ P3*(eta**3) + P4*(eta**4)


        mjd = mjd - tau
        flux = flux/(10**(-0.4*(poly))) 
        err=err/(10**(-0.4*(poly)))                    
        flux = (flux - B1)/(A1)
        err=err/A1

        


        if (add_var ==True):
            params.append([tau, P0, P1, P2, P3, P4, V])    
        else:
            params.append([tau, P0, P1, P2, P3, P4])    
            
            
            
        #Output fluxes with added variance
        if (add_var==False):
            V=0.0
        data_fluxes.append(np.transpose([data[i][:,0], data[i][:,1], np.sqrt((data[i][:,2]**2) + (V**2))]))
                    

        #Add shifted data to merged lightcurve        
        for j in range(len(mjd)):
            merged_mjd.append(mjd[j])
            merged_flux.append(flux[j])
            merged_err.append(err[j])
                
    merged_mjd = np.array(merged_mjd)
    merged_flux = np.array(merged_flux)
    merged_err = np.array(merged_err)
    

    delta = np.percentile(samples_chunks[-1][2], [16, 50, 84])[1]
    params.append([A1, B1, delta])
    
            
    params = list(chain.from_iterable(params))#Flatten into single array
    params=np.delete(params, [0,1, 2, 3, 4, 5])    

    #Calculate ROA to merged lc
    t, m, errs = RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
    #Normalise lightcurve
    m_mean = np.mean(m)#np.average(m, weights = 1.0/(errs**2))
    m_rms = np.std(m)
    t, m, errs = RunningOptimalAverageOutp(merged_mjd, merged_flux, merged_err, delta)
    m = (m-m_mean)/m_rms
    errs = errs/m_rms
    #Remove first tau
    print("Best Fit Parameters")
    table = [params]
    print(tabulate(table, headers=labels))
    

    #Write samples to file
    filehandler = open(file+"samples_flat.obj","wb")
    pickle.dump(samples_flat,filehandler)
    filehandler = open(file +"samples.obj","wb")
    pickle.dump(samples,filehandler)
    filehandler = open(file +"X_t.obj","wb")
    pickle.dump([t, m, errs],filehandler)       
    
    plt.rcParams.update({'font.size': 15})
    #Save Cornerplot to figure
    fig = corner.corner(samples_flat, labels=labels_latex, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 20});
    i = 0
    while os.path.exists('{}{:d}.pdf'.format("CornerPlot", i)):
        i += 1
    fig.savefig('{}{:d}.pdf'.format("CornerPlot", i))
    plt.close();
      
    
    
    return samples, samples_flat, t, m, errs, data_fluxes




def FluxToMag(flux, flux_err,flux_convert_factor):
    mag = -2.5*np.log10(flux/(flux_convert_factor))
    mag_err=1.0857*flux_err/flux
    return mag, mag_err


def MagToFlux(mag, mag_err,flux_convert_factor):
    flux =(flux_convert_factor)*10**(-0.4*mag)  
    flux_err = mag_err*flux/1.0857
    return flux, flux_err

class GravLensFit():
    def __init__(self, datadir, objName, images, priors, init_tau = None, init_delta=10.0, add_var=True, sig_level = 4.0, Nsamples=10000, Nburnin=5000, flux_convert_factor=3.0128e-5):
        self.datadir=datadir
        self.objName=objName
        self.images=images
        data=[]
        for i in range(len(images)):
            file = datadir + str(self.objName) +"_"+ str(self.images[i]) + ".dat"
            data.append(np.loadtxt(file))
            


        
        self.priors= priors
        self.init_tau = init_tau
        self.init_delta=init_delta
        self.add_var = add_var
        self.sig_level = sig_level
        self.Nsamples = Nsamples
        self.Nburnin = Nburnin
        self.flux_convert_factor=flux_convert_factor
        if (init_tau == None):
            self.init_tau = [0]*len(data)
        else:
            self.init_tau = init_tau
            
            
        #Covert to flux
       # print("Converting to Flux")
        
        data_L = []
        for i in range(len(data)):
            err=[]
            mjd = data[i][:,0]
            flux, flux_err = MagToFlux(np.array(data[i][:,1]), np.array(data[i][:,2]),self.flux_convert_factor)
            data_L.append(np.transpose([mjd, flux, flux_err])) 
        
        
        run = LensFit(data_L, self.priors, self.init_tau, self.init_delta, self.add_var, self.sig_level, self.Nsamples, self.Nburnin, self.images, self.datadir)

        self.samples = run[0]
        self.samples_flat = run[1]
        self.t = run[2]
        self.X = run[3]
        self.X_errs = run[4]
        self.data_fluxes = run[5] # With extra errors added
        
        #Convert back to magnitudes for plotting
       # print("Converting back to magnitudes for plotting")
        data_mag=[]
        for i in range(len(data)):
            err=[]
            mjd = self.data_fluxes[i][:,0]
            mag, mag_err = FluxToMag(self.data_fluxes[i][:,1], self.data_fluxes[i][:,2],self.flux_convert_factor)
            data_mag.append(np.transpose([mjd, mag, mag_err]))
            
                
        plt.style.use(['science','no-latex'])        
        plt.rcParams.update({
            "font.family": "Sans", 
            "font.serif": ["DejaVu"],
            "figure.figsize":[15,20],
            "font.size": 20,
            "xtick.major.size" : 6,
            "xtick.major.width": 1.2,
            "xtick.minor.size" : 3,
            "xtick.minor.width" : 1.2,
            "ytick.major.size" : 6,
            "ytick.major.width": 1.2,
            "ytick.minor.size" : 3,
            "ytick.minor.width" : 1.2})      
            
            

        #Plotting
        fig = plt.figure(100)
        
        height_ratios=[1, 0.5]*len(data)
        gs = fig.add_gridspec(int(2.0*len(data_mag)), 1, hspace=0, wspace=0, height_ratios = height_ratios)
        axs = gs.subplots(sharex='col') 
        
        if (add_var == True):
            chunk_size=int(7)
        else:
            chunk_size=int(6)
            
        transpose_samples = np.transpose(self.samples_flat)
        transpose_samples = np.insert(transpose_samples, [0], np.array([0.0]*len(transpose_samples[1])), axis=0)
        transpose_samples = np.insert(transpose_samples, [1], np.array([0.0]*len(transpose_samples[1])), axis=0)    
        transpose_samples = np.insert(transpose_samples, [2], np.array([0.0]*len(transpose_samples[1])), axis=0)    
        transpose_samples = np.insert(transpose_samples, [3], np.array([0.0]*len(transpose_samples[1])), axis=0)
        transpose_samples = np.insert(transpose_samples, [4], np.array([0.0]*len(transpose_samples[1])), axis=0)    
        transpose_samples = np.insert(transpose_samples, [5], np.array([0.0]*len(transpose_samples[1])), axis=0)   
        
        
        samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)]

        A0 = np.percentile(samples_chunks[-1][0], [16, 50, 84])[1]   
        B0 = np.percentile(samples_chunks[-1][1], [16, 50, 84])[1] 
        band_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for j in range(len(data_mag)):
            for i in range(2):
                #Read in parameters
                tau = np.percentile(samples_chunks[j][0], [16, 50, 84])[1]
                P0 = np.percentile(samples_chunks[j][1], [16, 50, 84])[1]
                P1 = np.percentile(samples_chunks[j][2], [16, 50, 84])[1]
                P2 = np.percentile(samples_chunks[j][3], [16, 50, 84])[1]
                P3 = np.percentile(samples_chunks[j][4], [16, 50, 84])[1]
                P4 = np.percentile(samples_chunks[j][5], [16, 50, 84])[1]

                
                #Read in data
                mjd=data_mag[j][:,0]
                flux=data_mag[j][:,1]
                flux_err=data_mag[j][:,2]
                #Calculate normalised time
                eta = (2.0*self.t -(max(mjd) + min(mjd)))/(max(mjd) - min(mjd))                     
                
                #Calculate model
                t_shifted = self.t + tau
                interp = interpolate.interp1d(t_shifted, self.X, kind="linear", fill_value="extrapolate")
                m_shifted = interp(self.t)                  
                m_scaled = (10**(-0.4*(P0 + P1*(eta**1) + P2*(eta**2)+ P3*(eta**3)+ P4*(eta**4))))*(A0*m_shifted + B0)
                
                #Calculate error envelope
                interp = interpolate.interp1d(t_shifted, self.X + self.X_errs, kind="linear", fill_value="extrapolate")
                m_shifted = interp(self.t)              
                m_scaled_up = (10**(-0.4*(P0 + P1*(eta**1) + P2*(eta**2)+ P3*(eta**3)+ P4*(eta**4))))*(A0*m_shifted + B0)          
                interp = interpolate.interp1d(t_shifted, self.X - self.X_errs, kind="linear", fill_value="extrapolate")
                m_shifted = interp(self.t)              
                m_scaled_down = (10**(-0.4*(P0 + P1*(eta**1) + P2*(eta**2)+ P3*(eta**3)+ P4*(eta**4))))*(A0*m_shifted + B0)                
                
                length=abs(max(mjd)-min(mjd))

                if (i==0):
                    z=int(2.0*j)
                    axs[z].errorbar(mjd, flux, yerr=flux_err, ls='none', marker=".", color=band_colors[j])
                    #Plot Model
                    mag = -2.5*np.log10(m_scaled/self.flux_convert_factor)
                    
                    mag_up=np.empty(len(self.t))
                    mag_down = np.empty(len(self.t))
            
                    for k in range(len(self.t)):
                        if (m_scaled_up[k]/self.flux_convert_factor>0):
                            mag_up[k] = -2.5*np.log10(m_scaled_up[k]/self.flux_convert_factor)     
                        else:
                            mag_up[k] = 1e5
                                  
                        if (m_scaled_down[k]/self.flux_convert_factor>0):                         
                            mag_down[k] = -2.5*np.log10(m_scaled_down[k]/self.flux_convert_factor)
                        else:
                            mag_down[k]=1e5

                                                         
                    axs[z].plot(self.t , mag, color="black")
                    axs[z].fill_between(self.t, mag_up, mag_down,facecolor="darkgrey", edgecolor='none', rasterized=True, antialiased=True)
                    
                    axs[z].set_ylabel(str(images[j])+"\n Mag")
                    axs[z].set_xlabel("MJD")           
                    axs[z].annotate("Image "+images[j], xy=(0.85, 0.85), xycoords='axes fraction', size=15.0,color=band_colors[j], fontsize=20)  
                    length=abs(max(flux)-min(flux))
                    axs[z].set_ylim(min(flux)-0.2*length, max(flux)+0.2*length)
                    axs[z].invert_yaxis() 
                    length=abs(max(mjd)-min(mjd))        
                    axs[z].set_xlim(min(mjd)-0.1*length, max(mjd)+0.1*length)            
                else:
                    fluxes = (self.flux_convert_factor)*10**(-0.4*data_mag[j][:,1])
                    t_shifted = self.t + tau
                    interp = interpolate.interp1d(t_shifted, self.X, kind="linear", fill_value="extrapolate")
                    mj = interp(mjd)
                      
            
                    #Find mag diff
                    delta_m = -2.5*np.log10((fluxes)/(A0*mj + B0))
                    
                    z=int(2.0*j+1.0)
            
                    axs[z].errorbar(mjd, delta_m, yerr=flux_err, ls='none', marker=".", color=band_colors[j])
                    axs[z].plot(self.t, P0 + P1*(eta**1) + P2*(eta**2)+ P3*(eta**3)+ P4*(eta**4), color="black")            
                    axs[z].set_xlabel("MJD")        
                    axs[z].set_ylabel("\u0394m")     
                    length=abs(max(delta_m)-min(delta_m))
                    axs[z].set_ylim(min(delta_m)-0.2*length, max(delta_m)+0.2*length)
                    axs[z].invert_yaxis()                  
        axs[0].set_title(self.objName)
        fig.show()
        i = 0
        while os.path.exists('{}{:d}.pdf'.format("GravLensPlot", i)):
            i += 1
        fig.savefig('{}{:d}.pdf'.format("GravLensPlot", i))
        #plt.close()      
        
        
        
        
        
        
        
        
        

