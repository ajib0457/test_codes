import numpy as np
#import h5py
import sys
sys.path.insert(0, '/project/GAMNSCM2/funcs') 
from plotter_funcs import *
from error_func import *
import pickle

sim_sz=500           #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250       #Density Field grid resolution
smooth_scl=3.5         #Smoothing scale in physical units Mpc/h
mass_bins=0          #Mass bin 
particles_filt=100   #Halos to filter out based on number of particles
lss_type=2           #Cluster-3 Filament-2 Sheet-1 Void-0
#MCMC initial cond.
ndim, nwalkers=1,10 #ndim-num. of dimentions of paramter space. nwalkers- amount of walkers
initial_c=0.4        #initial value of what c-correlation coeffiient could be
burn_in=50          #how many steps are disposed of
steps_wlk=70       #number of steps for each walker taken
#plotting features
bins=500             #the posterior histogram bins
sigma_area=0.3413#sigma percentile area as decimal
snapshot=11                #'12  '11'
cosmology='lcdm'           #'lcdm'  'cde0'  'wdm2'
tot_mass_bins=5

filehandler = open('/scratch/GAMNSCM2/%s/snapshot_0%s/correl/my_den/files/output_files/dotproduct/spin_lss/%s_snapshot_0%s_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(cosmology,snapshot,cosmology,snapshot,lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"rb")
diction=pickle.load(filehandler)
filehandler.close()

results=diction['results']
del diction['results']
c_samples={}#dictionary for output data 

dot_val=np.asarray(diction[mass_bins])
c_samples[mass_bins]=mcmc(mass_bins,dot_val,initial_c,nwalkers,ndim,burn_in,steps_wlk)

#Redefine posterior into scatter plot of posterior
a=np.histogram(c_samples[mass_bins],density=True,bins=bins)
x=a[1]
dx=(np.max(x)-np.min(x))/bins
x=np.delete(x,len(x)-1,0)+dx/2
y=a[0]
 
perc_lo,results[mass_bins,4],perc_hi,results[mass_bins,3],results[mass_bins,2],area=error(x,y,dx,sigma_area)

#posterior_plt(cosmology,c_samples,results,bins,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method,grid_density=grid_density,no_halos=no_halos)    
#
#mod_data_ovrplt(cosmology,diction,results,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,dp_bins=15)

filehandler = open('/scratch/GAMNSCM2/%s/snapshot_0%s/correl/my_den/files/output_files/mod_fit/mcmc_mthd/myden_mcmcresults_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(cosmology,snapshot,lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"wb")
pickle.dump(results,filehandler)
filehandler.close() 
