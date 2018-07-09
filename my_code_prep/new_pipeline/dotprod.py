import numpy as np
import math as mth
import h5py
import sklearn.preprocessing as skl 
import sys
sys.path.insert(0, '/project/GAMNSCM2/funcs') 
from plotter_funcs import *
from error_func import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from decimal import Decimal

sim_type=sys.argv[1]          #'dm_only' 'DTFE'
cosmology=sys.argv[2]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot=sys.argv[3]      #'12  '11'...
den_type=sys.argv[4]           #'DTFE' 'my_den'
smooth_scl=Decimal(sys.argv[5])#Smoothing scale in physical units Mpc/h. 2  3.5  5
tot_mass_bins=int(sys.argv[6])      #Number of Halo mass bins
particles_filt=int(sys.argv[7])   #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
total_lss_parts=int(sys.argv[8])    #Total amount of lss_class parts

sim_sz=500           #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250      #Density Field grid resolution
lss_type=2           #Cluster-3 Filament-2 Sheet-1 Void-0
runs=100000          #bootstrap resampling runs
method='bootstrap'   #Not optional for this code
hist_bins=200        #histogram to form gaussian pdf
sigma_area=0.3413    #sigma percentile area as decimal

recon_vecs_x=np.zeros((grid_nodes**3))
recon_vecs_y=np.zeros((grid_nodes**3))
recon_vecs_z=np.zeros((grid_nodes**3))
mask=np.zeros((grid_nodes**3))
for part in range(total_lss_parts):

    nrows_in=int(1.*(grid_nodes**3)/total_lss_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/total_lss_parts)
    f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/eigvecs/%s_sim%s_recon_vecs_sim%s_smth%sMpc_gd%d_%d.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes,part), 'r')
    recon_vecs_x[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
    f2=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/eigvecs/%s_sim%s_recon_vecs_sim%s_smth%sMpc_gd%d_%d_mask.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes,part), 'r')
    mask[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()

#f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_box_%scubed_xyz_vxyz_jxyz_m_r.h5"%sim_sz, 'r')#xyz vxvyvz jxjyjz & Rmass & Rvir: Halo radius (kpc/h comoving).
f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/catalogs/%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart.h5"%(sim_type,cosmology,snapshot,sim_type,cosmology,snapshot), 'r')
data=f['/halo'][:]#data array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(Mpc/h)*km/s), (Vir. Mass)Mvir(Msun/h) & (Vir. Rad)Rvir(kpc/h) 
f.close()
#Prebinning for dotproduct binning within loop ------
Xc_min=np.min(data[:,0])
Xc_max=np.max(data[:,0])
Yc_min=np.min(data[:,1])
Yc_max=np.max(data[:,1])
Zc_min=np.min(data[:,2])
Zc_max=np.max(data[:,2])

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001
#----------------------------------------------------
recon_vecs_flt_unnorm=np.column_stack((recon_vecs_x,recon_vecs_y,recon_vecs_z))
del recon_vecs_x
del recon_vecs_y
del recon_vecs_z
mask=np.reshape(mask,(grid_nodes,grid_nodes,grid_nodes))

recon_vecs=np.reshape(recon_vecs_flt_unnorm,(grid_nodes,grid_nodes,grid_nodes,3))#Reshape eigenvectors

#'data' format reminder: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(Mpc/h)*km/s), (Vir. Mass)Mvir(Msun/h) & (Vir. Rad)Rvir(kpc/h)
#partcl_halo_flt=np.where((data[:,9]/(Mass_res))>=particles_filt)#filter for halos with <N particles
partcl_halo_flt=np.where(data[:,11]>=particles_filt)#filter for halos with <N particles
data=data[partcl_halo_flt]#Filter out halos with <N particles

#apend eigvecs on to data array and apend mask value to data array
#create empty arrays and apend to data array
color_cd=np.zeros((len(data),1))
norm_eigvecs=np.zeros((len(data),3))
data=np.hstack((data,color_cd,norm_eigvecs))
for i in range(len(data)):
   #Create index from halo coordinates
    grid_index_x=mth.trunc(data[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(data[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(data[i,2]*Zc_mult-Zc_minus) 
    data[i,12]=mask[grid_index_x,grid_index_y,grid_index_z]
    data[i,13:16]=recon_vecs[grid_index_x,grid_index_y,grid_index_z,:]
del recon_vecs
del mask
#NEW 'data' format :(Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(Mpc/h)*km/s), (Vir. Mass)Mvir(Msun/h), 
#(Vir. Rad)Rvir(kpc/h), mask(0,1,2,3) & eigvecs(ex,ey,ez) 
fil_filt=np.where(data[:,12]==lss_type)#2-filament
data=data[fil_filt]
  
halo_mass=data[:,9]
log_halo_mass=np.log10(halo_mass)#convert into log10(M)
mass_intvl=(np.max(log_halo_mass)-np.min(log_halo_mass))/tot_mass_bins#log_mass value used to find mass interval
diction={}
diction_2={}#This dictionary contains all of the means of all of the samples taken from the dataset of dp's.
results=np.zeros((tot_mass_bins,5))# [Mass_min, Mass_max, Value, Error+,Error-] 
for mass_bin in range(tot_mass_bins):
    
    low_int_mass=np.min(log_halo_mass)+mass_intvl*mass_bin#Calculate mass interval
    hi_int_mass=low_int_mass+mass_intvl+0.000000001#Calculate mass interval
    results[mass_bin,0]=low_int_mass#Store mass interval
    results[mass_bin,1]=hi_int_mass#Store mass interval   
    #Create mask to filter out halos within mass interval
    mass_mask=np.logical_and(log_halo_mass>=low_int_mass,log_halo_mass<hi_int_mass)
    data_mass_bin=data[mass_mask]

    #normalize vectors
    data_mass_bin[:,13:16]=skl.normalize(data_mass_bin[:,13:16])#Eigenvectors
    data_mass_bin[:,6:9]=skl.normalize(data_mass_bin[:,6:9])#Halo AM
    store_dp=np.zeros(len(data_mass_bin))
    for i in range(len(data_mass_bin)):
        store_dp[i]=np.inner(data_mass_bin[i,13:16],data_mass_bin[i,6:9])#take the dot product between vecs, row by row   
    store_dp=abs(store_dp)    
    diction[mass_bin]=store_dp    
    
    #Calculating error using bootstrap resampling
    a=np.random.randint(low=0,high=len(store_dp),size=(runs,len(store_dp)))
    mean_set=np.mean(store_dp[a],axis=1)
    
    a=np.histogram(mean_set,density=True,bins=hist_bins)
    x=a[1]
    dx=(np.max(x)-np.min(x))/hist_bins
    x=np.delete(x,len(x)-1,0)+dx/2
    y=a[0]
    perc_lo,results[mass_bin,4],perc_hi,results[mass_bin,3],results[mass_bin,2],area=error(x,y,dx,sigma_area)
    
    diction_2[mass_bin]=mean_set
         
    diction['results']=results
'''       
f=h5py.File('/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/dotproduct/spin_lss/myden_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.h5'%(lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),'w')     
for mass_bins in range(tot_mass_bins):   
    f.create_dataset('/dp%s'%mass_bins,data=dict[mass_bins])
f.create_dataset('/results',data=results)
f.close()
'''

filehandler = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/dotproduct/spin_lss/%s_snapshot_0%s_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s.pkl'%(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,lss_type,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt),"wb")
pickle.dump(diction,filehandler)    
#Plot correlation    
alignment_plt(grid_nodes,results,smooth_scl,sim_type,cosmology,snapshot,den_type)
  
posterior_plt(cosmology,diction_2,results,hist_bins,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type)
 
