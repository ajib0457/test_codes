def error(x,y,dx,sigma_area):
    import numpy as np
    
    y=y/(np.sum(y)*dx)#normalize posterior
    log_max_ind=np.where(y==np.max(y))[0]#Find peak percentile
    most_lkl=x[log_max_ind[0]]#take the zero out since this is only needed for toy example since 
    #this is a perfect gaussian thus symmetrial and will have 2 maximum values, thus chose first
    area=np.sum(y[0:log_max_ind[0]])*dx
    
    def sigma_finder(most_lkl,percentile,err_type):
        b=0
        sig_ind=0
        while b<percentile and b<1:
            sig_ind+=1
            b=np.sum(y[0:sig_ind])*dx+10**-15#10^-15 makes up for numerical error in calculating area
                 
        if err_type=='-':
            sig_c_m=most_lkl - x[sig_ind-1]
            
        if err_type=='+':
            sig_c_m=x[sig_ind-1] - most_lkl
    
        return b,sig_c_m
    
    #Run 1 sigma error calculations
    
    perc_lo,sig_c_lo=sigma_finder(most_lkl,percentile=area-sigma_area,err_type='-')
    perc_hi,sig_c_hi=sigma_finder(most_lkl,percentile=area+sigma_area,err_type='+')

    return perc_lo,sig_c_lo,perc_hi,sig_c_hi,most_lkl,area

def grid_mthd(dot_val,grid_density,sigma_area): 

    import numpy as np
    
    c,dc=np.linspace(-0.99,0.99,grid_density,retstep=True)
    loglike=np.zeros((len(c),1))
    for i in range(len(c)):
            
        loglike[i,0]=np.sum(np.log((1-c[i])*np.sqrt(1+(c[i]/2))*(1-c[i]*(1-1.5*(dot_val)**2))**(-1.5)))#log-likelihood
    
    #Convert loglike into likelihood function (real space)
    loglike_like=np.exp(loglike)
    
    perc_lo,sig_c_lo,perc_hi,sig_c_hi,most_lkl,area=error(c,loglike_like,dc,sigma_area)
    
    return loglike_like,most_lkl,sig_c_hi,sig_c_lo


def mcmc(mass_bins,dot_val,initial_c,nwalkers,ndim,burn_in,steps_wlk):
    import numpy as np
    import scipy.optimize as op
    import emcee
    from emcee.utils import MPIPool
    import sys
    
    def lnlike(c,dot_val):
        loglike=np.zeros((1))
        loglike[0]=sum(np.log((1-c)*np.sqrt(1+(c/2))*(1-c*(1-3*(dot_val*dot_val/2)))**(-1.5)))#log-likelihood 
    
        return loglike
        
    def lnprior(c):
        
        if (-1.5 < c < 0.99):#Assumes a flat prior, uninformative prior
            return 0.0
        return -np.inf
        
    def lnprob(c,dot_val):
        lp = lnprior(c)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(c,dot_val)
    #Parallel MCMC - initiallizes pool object; if process isn't running as master, wait for instr. and exit
    pool=MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    
    pos = [initial_c+1e-2*np.random.randn(ndim) for i in range(nwalkers)]#initial positions for walkers "Gaussian ball"
     
    #MCMC Running
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,args=[dot_val],pool=pool)
    
    pos, _, _=sampler.run_mcmc(pos,burn_in)#running of emcee burn-in period
    sampler.reset()
    
    sampler.run_mcmc(pos, steps_wlk)#running of emcee for steps specified, using pos as initial walker positions
    pool.close()
    chain=sampler.flatchain[:,0]  
    
    return chain
