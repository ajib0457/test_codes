def grid_mthd(mass_bins,dot_val,grid_density): 

    import numpy as np
    
    c=np.linspace(-0.99,0.99,grid_density)
    loglike=np.zeros((len(c),1))
    for i in range(len(c)):
            
        loglike[i,0]=np.sum(np.log((1-c[i])*np.sqrt(1+(c[i]/2))*(1-c[i]*(1-1.5*(dot_val)**2))**(-1.5)))#log-likelihood
    
    #Convert loglike into likelihood function (real space)
    loglike_like=np.exp(loglike)
    max_val_loglike=np.where(loglike==np.max(loglike))[0]#be careful that this condition avoids nan values within array
    c_value_loglike=c[max_val_loglike]
    
    #error calculation at full width half maximum
    FWHM_hf_hght=1.*(np.max(loglike_like)-np.min(loglike_like))/2
    FWHM_index=abs(abs(loglike_like)-FWHM_hf_hght).argmin()
    sig_c=c_value_loglike-c[FWHM_index]
    
    return loglike_like,c_value_loglike,sig_c
