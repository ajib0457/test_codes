import numpy as np
from grid_mthd_func import * 
import pickle
import sys
sys.path.insert(0, '/project/GAMNSCM2/') 
from plotter_funcs import *

sim_sz=250           #Size of simulation in physical units Mpc/h cubed
grid_nodes=850       #Density Field grid resolution
smooth_scl=2         #Smoothing scale in physical units Mpc/h
tot_mass_bins=5      #Number of Halo mass bins
particles_filt=500   #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
lss_type=2           #Cluster-3 Filament-2 Sheet-1 Void-0
#grid method initial cond.
grid_density=5000    #density c value parameter space ranging from -0.99 to 0.99
#plotting features
bins=100             #the posterior histogram bins

filehandler = open('/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/dotproduct/spin_lss/myden_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"rb")
diction=pickle.load(filehandler)
filehandler.close()

results=diction['results']
del diction['results']
c_samples={}#dictionary for output data 
 
for mass_bins in range(tot_mass_bins):
    
    dot_val=np.asarray(diction[mass_bins])
    c_samples[mass_bins],results[mass_bins,2],results[mass_bins,3]=grid_mthd(mass_bins,dot_val,grid_density)
    
'''
posterior_plt(c_samples,results,bins,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,nwalkers,steps_wlk)    
'''
fig_5_trowland(results,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type)

filehandler = open('/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/mod_fit/grid_mthd/myden_gridresults_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"wb")
pickle.dump(results,filehandler)
filehandler.close()  

