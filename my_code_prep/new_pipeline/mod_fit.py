import numpy as np
import scipy.optimize as op
import emcee
from emcee.utils import MPIPool
import sys
import h5py
import sys
sys.path.insert(0, '/project/GAMNSCM2/') 
from plotter_funcs import *

sim_sz=250           #Size of simulation in physical units Mpc/h cubed
grid_nodes=850       #Density Field grid resolution
smooth_scl=2         #Smoothing scale in physical units Mpc/h
tot_mass_bins=5      #Number of Halo mass bins
particles_filt=300   #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
lss_type=2           #Cluster-3 Filament-2 Sheet-1 Void-0
#MCMC initial cond.
ndim, nwalkers=1,300 #ndim-num. of dimentions of paramter space. nwalkers- amount of walkers
initial_c=0.4        #initial value of what c-correlation coeffiient could be
burn_in=500          #how many steps are disposed of
steps_wlk=3000       #number of steps for each walker taken
#plotting features
bins=700             #the posterior histogram bins

dict={}#dictionary for input data
f=h5py.File('/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/dotproduct/spin_lss/myden_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.h5'%(lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),'r')     
for mass_bins in range(tot_mass_bins):   
    dict[mass_bins]=f['/dp%s'%mass_bins][:]
results=f['/results'][:]# [Mass_min, Mass_max, Value, Error] 
f.close()

c_samples={}#dictionary for output data 
for mass_bins in range(tot_mass_bins):
    #MCMC
    dot_val=np.asarray(dict[mass_bins])
    
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
    c_samples[mass_bins]=sampler.flatchain[:,0]
    results[mass_bins,2]=round(np.mean(c_samples[mass_bins]),4)
    results[mass_bins,3]=round(np.std(c_samples[mass_bins],dtype=np.float64),4)

posterior_plt(c_samples,results,bins,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,nwalkers,steps_wlk)    

'''   
def fig_5_trowland(results):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    #Trowland et al. 2012 data -------------------------------------------------
    mass_holly=np.array([11.63,12.21,12.79,13.37,13.95])
    #Holly's figure-5 3.5Mpc scale, z=0 (red)
    c_vals_holly_3=np.array([-0.044,-0.00001,0.052,0.145,0.462])
    c_error_holly_neg_3=c_vals_holly_3-np.array([-0.04,-0.01,0.048,0.1,0.15])
    c_error_holly_pos_3=np.array([-0.048,0.005,0.075,0.2,0.55])-c_vals_holly_3
    #Holly's figure-5 2Mpc scale, z=0 (red)
    c_vals_holly_2=np.array([-0.047,-0.023,0.025,0.145,0.057])
    c_error_holly_neg_2=c_vals_holly_2-np.array([-0.05,-0.04,0.0001,-0.05,-0.2])
    c_error_holly_pos_2=np.array([-0.058,-0.005,0.05,0.2,0.3])-c_vals_holly_2
    #Holly's figure-5 5Mpc scale, z=0 (red)
    c_vals_holly_5=np.array([-0.044,0.003,0.04,0.14,0.27])
    c_error_holly_neg_5=c_vals_holly_5-np.array([-0.047,-0.004,0.03,0.065,0.05])
    c_error_holly_pos_5=np.array([-0.043,0.0025,0.055,0.17,0.4])-c_vals_holly_5
    #---------------------------------------------------------------------------

    plt.figure()
    
    ax2=plt.subplot2grid((1,1), (0,0))
    ax2.axhline(y=0, xmin=0, xmax=15, color = 'k',linestyle='--')
    
    ax2.plot(results[:,0],results[:,2],'g-',label='halo_LSS. 3.5Mpc/h')
    ax2.fill_between(results[:,0], results[:,2]-results[:,3], results[:,2]+results[:,3],facecolor='green',alpha=0.3)
    
    plt.ylabel('Mean cos(theta)')
    plt.xlabel('log Mass[M_solar]')   
    plt.title('Spin-Filament')
    plt.legend(loc='upper right')
    plt.savefig('ALIGNMENT_PLOT_grid_%s_smth_scl%s.png'%(grid_nodes,smooth_scl))
'''
