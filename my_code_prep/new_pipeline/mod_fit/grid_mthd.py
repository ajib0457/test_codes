import numpy as np
import pickle
import sys
sys.path.insert(0, '/project/GAMNSCM2/funcs') 
from plotter_funcs import *
from error_func import *
from decimal import Decimal

sim_sz=500                    #Size of simulation in physical units Mpc/h cubed
sim_type=sys.argv[1]          #'dm_only' 'DTFE'
cosmology=sys.argv[2]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot=sys.argv[3]      #'12  '11'...
den_type=sys.argv[4]           #'DTFE' 'my_den'
smooth_scl=Decimal(sys.argv[5])#Smoothing scale in physical units Mpc/h. 2  3.5  5
tot_mass_bins=int(sys.argv[6])      #Number of Halo mass bins
particles_filt=int(sys.argv[7])   #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
grid_nodes=1250      #Density Field grid resolution
lss_type=2           #Cluster-3 Filament-2 Sheet-1 Void-0
grid_density=11000   #density c value parameter space ranging from -0.99 to 0.99
bins=100             #For plotting: the posterior histogram bins
method='grid'        #not an option for this code
sigma_area=0.3413    #sigma percentile area as decimal



filehandler = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/dotproduct/spin_lss/%s_snapshot_0%s_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"rb")
diction=pickle.load(filehandler)
filehandler.close()

results=diction['results']
del diction['results']
c_samples={}#dictionary for output data 
no_halos=np.zeros((tot_mass_bins,1)) 
for mass_bins in range(tot_mass_bins):
    
    dot_val=np.asarray(diction[mass_bins])
    no_halos[mass_bins]=len(dot_val)
    c_samples[mass_bins],results[mass_bins,2],results[mass_bins,3],results[mass_bins,4]=grid_mthd(dot_val,grid_density,sigma_area)
    

posterior_plt(cosmology,c_samples,results,bins,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,grid_density=grid_density,no_halos=no_halos)     

fig_5_trowland(cosmology,snapshot,results,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method,sim_type,den_type)

mod_data_ovrplt(cosmology,diction,results,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,dp_bins=15)

filehandler = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/mod_fit/grid_mthd/myden_gridresults_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(sim_type,cosmology,snapshot,den_type,lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"wb")
pickle.dump(results,filehandler)
filehandler.close()  

