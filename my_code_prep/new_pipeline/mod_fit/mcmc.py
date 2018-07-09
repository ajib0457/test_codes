import numpy as np
#import h5py
import sys
sys.path.insert(0, '/project/GAMNSCM2/funcs') 
from plotter_funcs import *
from error_func import *
import pickle
import scipy.optimize as op
import emcee
from emcee.utils import MPIPool
from decimal import Decimal

mass_bins=int(sys.argv[1])-1  #Mass bin
sim_type=sys.argv[2]          #'dm_only' 'DTFE'
cosmology=sys.argv[3]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot=sys.argv[4]      #'12  '11'...
den_type=sys.argv[5]           #'DTFE' 'my_den'
smooth_scl=Decimal(sys.argv[6])#Smoothing scale in physical units Mpc/h. 2  3.5  5
tot_mass_bins=int(sys.argv[7])      #Number of Halo mass bins
particles_filt=int(sys.argv[8])   #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
#MCMC initial cond.
ndim, nwalkers=1,int(sys.argv[9])         #ndim-num. of dimentions of paramter space. nwalkers- amount of walkers
burn_in=int(sys.argv[10])                   #how many steps are disposed of from the walker chain
steps_wlk=int(sys.argv[11])                #number of steps for each walker taken
initial_c=Decimal(sys.argv[12])               #initial value of what c-correlation coeffiient could be

sim_sz=500                    #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250               #Density Field grid resolution
lss_type=2                    #Cluster-3 Filament-2 Sheet-1 Void-0
bins=500                      #Plotting: the posterior histogram bins
sigma_area=0.3413             #sigma percentile area as decimal

filehandler = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/dotproduct/spin_lss/%s_snapshot_0%s_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"rb")
diction=pickle.load(filehandler)
filehandler.close()

c_results={}
results=np.zeros(5)
results[0:2]=diction['results'][mass_bins,0:2]

dot_val=np.asarray(diction[mass_bins])
    
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
c_results[mass_bins]=chain   
 
#Redefine posterior into scatter plot of posterior
a=np.histogram(chain,density=True,bins=bins)
x=a[1]
dx=(np.max(x)-np.min(x))/bins
x=np.delete(x,len(x)-1,0)+dx/2
y=a[0]
 
perc_lo,results[4],perc_hi,results[3],results[2],area=error(x,y,dx,sigma_area)
c_results['results']=results
filehandler = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/mod_fit/mcmc_mthd/myden_mcmcresults_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(sim_type,cosmology,snapshot,den_type,lss_type,sim_sz,grid_nodes,smooth_scl,mass_bins,particles_filt),"wb")
pickle.dump(c_results,filehandler)
filehandler.close() 
