import pickle
import numpy as np
import sys
sys.path.insert(0, '/project/GAMNSCM2/funcs') 
from plotter_funcs import *
from error_func import *

sim_sz=500           #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250      #Density Field grid resolution
smooth_scl=3.5       #Smoothing scale in physical units Mpc/h
mass_bins=0          #Mass bin 
particles_filt=100   #Halos to filter out based on number of particles
lss_type=2           #Cluster-3 Filament-2 Sheet-1 Void-0
#MCMC initial cond.
ndim, nwalkers=1,10  #ndim-num. of dimentions of paramter space. nwalkers- amount of walkers
initial_c=0.4        #initial value of what c-correlation coeffiient could be
burn_in=50           #how many steps are disposed of
steps_wlk=70         #number of steps for each walker taken
#plotting features
bins=500             #the posterior histogram bins
sigma_area=0.3413#sigma percentile area as decimal
snapshot=11                #'12  '11'
cosmology='lcdm'           #'lcdm'  'cde0'  'wdm2'
tot_mass_bins=5
method='mcmc'

c_results={}#used only within the for loop
c_chain={}#stores c chain
results=np.zeros((tot_mass_bins,5))# [Mass_min, Mass_max, Value, Error+,Error-]
for mass_bins in range(tot_mass_bins):
    filehandler = open('/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/correl/my_den/files/output_files/mod_fit/mcmc_mthd/myden_mcmcresults_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(cosmology,snapshot,lss_type,sim_sz,grid_nodes,smooth_scl,mass_bins,particles_filt),"rb")
    c_results=pickle.load(filehandler)
    results[mass_bins,:]=c_results['results']    
    c_chain[mass_bins]=c_results[mass_bins]    
    filehandler.close()

filehandler = open('/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/correl/my_den/files/output_files/dotproduct/spin_lss/%s_snapshot_0%s_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(cosmology,snapshot,cosmology,snapshot,lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"rb")
diction=pickle.load(filehandler)
filehandler.close() 
    
posterior_plt(cosmology,c_chain,results,bins,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method) 
    
mod_data_ovrplt(cosmology,diction,results,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method,dp_bins=15)

fig_5_trowland(cosmology,snapshot,results,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method)
