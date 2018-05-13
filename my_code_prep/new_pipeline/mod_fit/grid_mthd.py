import numpy as np
from grid_mthd_func import * 
import pickle
import sys
sys.path.insert(0, '/project/GAMNSCM2/funcs') 
from plotter_funcs import *

sim_sz=500           #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250       #Density Field grid resolution
smooth_scl=3.5         #Smoothing scale in physical units Mpc/h
tot_mass_bins=5      #Number of Halo mass bins
particles_filt=100   #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
lss_type=2           #Cluster-3 Filament-2 Sheet-1 Void-0
#grid method initial cond.
grid_density=5000    #density c value parameter space ranging from -0.99 to 0.99
#plotting features
bins=100             #the posterior histogram bins
cosmology='cde0'           #'lcdm'  'cde0'  'wdm2'
snapshot=11                #'12  '11'
method='grid'

filehandler = open('/scratch/GAMNSCM2/%s/snapshot_0%s/correl/my_den/files/output_files/dotproduct/spin_lss/%s_snapshot_0%s_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(cosmology,snapshot,cosmology,snapshot,lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"rb")
diction=pickle.load(filehandler)
filehandler.close()

results=diction['results']
del diction['results']
c_samples={}#dictionary for output data 
 
for mass_bins in range(tot_mass_bins):
    
    dot_val=np.asarray(diction[mass_bins])
    c_samples[mass_bins],results[mass_bins,2],results[mass_bins,3]=grid_mthd(mass_bins,dot_val,grid_density)
    

posterior_plt(cosmology,c_samples,results,bins,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method,grid_density=grid_density)    

fig_5_trowland(results,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type)

mod_data_ovrplt(cosmology,diction,results,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,dp_bins=15)

filehandler = open('/scratch/GAMNSCM2/%s/snapshot_0%s/correl/my_den/files/output_files/mod_fit/grid_mthd/myden_gridresults_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(cosmology,snapshot,lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"wb")
pickle.dump(results,filehandler)
filehandler.close()  
