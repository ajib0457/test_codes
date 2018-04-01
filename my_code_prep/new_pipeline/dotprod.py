import numpy as np
import math as mth
import h5py
import sklearn.preprocessing as skl 
from plotter_funcs import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sim_sz=250           #Size of simulation in physical units Mpc/h cubed
grid_nodes=850      #Density Field grid resolution
smooth_scl=3.5      #Smoothing scale in physical units Mpc/h
tot_mass_bins=5     #Number of Halo mass bins
particles_filt=500  #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
Mass_res=1.35*10**8 #Bolchoi particle mass as per, https://arxiv.org/pdf/1002.3660.pdf
total_lss_parts=8   #Total amount of lss_class parts

recon_vecs_x=np.zeros((grid_nodes**3))
recon_vecs_y=np.zeros((grid_nodes**3))
recon_vecs_z=np.zeros((grid_nodes**3))
mask=np.zeros((grid_nodes**3))
for part in range(total_lss_parts):

    nrows_in=int(1.*(grid_nodes**3)/total_lss_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/total_lss_parts)
    f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_sim%s_recon_vecs_DTFE_gd%d_smth%sMpc_%d.h5" %(sim_sz,grid_nodes,smooth_scl,part), 'r')
    recon_vecs_x[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
    f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_sim%s_recon_vecs_DTFE_gd%d_smth%sMpc_%d_mask.h5" %(sim_sz,grid_nodes,smooth_scl,part), 'r')
    mask[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()

#f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_box_%scubed_xyz_vxyz_jxyz_m_r.h5"%sim_sz, 'r')#xyz vxvyvz jxjyjz & Rmass & Rvir: Halo radius (kpc/h comoving).
f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_allhalos_xyz_vxyz_jxyz_m_r.h5", 'r')
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

recon_vecs_flt_norm=skl.normalize(recon_vecs_flt_unnorm)#normalize eigenvectors to make sure.
del recon_vecs_flt_unnorm
recon_vecs=np.reshape(recon_vecs_flt_norm,(grid_nodes,grid_nodes,grid_nodes,3))#Reshape eigenvectors

partcl_halo_flt=np.where((data[:,9]/(Mass_res))>=particles_filt)#filter for halos with <N particles
data=data[partcl_halo_flt]#Filter out halos with <N particles
halo_mass=data[:,9]
log_halo_mass=np.log10(halo_mass)#convert into log10(M)
mass_intvl=(np.max(log_halo_mass)-np.min(log_halo_mass))/tot_mass_bins#log_mass value used to find mass interval

results=np.zeros((tot_mass_bins,4))# [Mass_min, Mass_max, Value, Error] 
for mass_bin in range(tot_mass_bins):
    
    #Positions
    Xc=data[:,0]#I must redefine these at each iteration since I am filtering them for each mass interval
    Yc=data[:,1]
    Zc=data[:,2]
    #Angular momenta
    Lx=data[:,6]
    Ly=data[:,7]
    Lz=data[:,8]

    low_int_mass=np.min(log_halo_mass)+mass_intvl*mass_bin#Calculate mass interval
    hi_int_mass=low_int_mass+mass_intvl#Calculate mass interval
    results[mass_bin,0]=low_int_mass#Store mass interval
    results[mass_bin,1]=hi_int_mass#Store mass interval
   
    #Create mask to filter out halos within mass interval
    mass_mask=np.zeros(len(log_halo_mass))
    loint=np.where(log_halo_mass>=low_int_mass)
    hiint=np.where(log_halo_mass<hi_int_mass)
    mass_mask[loint]=1
    mass_mask[hiint]=mass_mask[hiint]+1
    mass_indx=np.where(mass_mask==2)
    
    #Positions & Angular Momenta filtered for mass interval
    Xc=Xc[mass_indx]
    Yc=Yc[mass_indx]
    Zc=Zc[mass_indx]
    halos_mom=np.column_stack((Lx,Ly,Lz))
    halos_mom=halos_mom[mass_indx]
    
    norm_halos_mom=skl.normalize(halos_mom)#Normalize Angular Momenta
    halos=np.column_stack((Xc,Yc,Zc,norm_halos_mom))
       
    store_spin=[]#Store Dot Product of spin-LSS
    for i in range(len(Xc)):
       #Create index from halo coordinates
        grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
        grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
        grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus) 
        
        if (mask[grid_index_x,grid_index_y,grid_index_z]==2):#Calculate Dot Product for filament-2
            spin_dot=np.inner(halos[i,3:6],recon_vecs[grid_index_x,grid_index_y,grid_index_z,:]) 
            store_spin.append(spin_dot)

    store_spin=np.asarray(store_spin) 
    costheta=abs(store_spin)#Take absolute value as only Alignment counts.
    if len(costheta)>0:      
        results[mass_bin,2]=np.mean(costheta)#Alignment value calc. and store
        
        #Calculating error using bootstrap resampling
        runs=300+mass_bin*300
        a=np.random.randint(low=0,high=len(costheta),size=(runs,len(costheta)))
        mean_set=np.mean(costheta[a],axis=1)
        results[mass_bin,3]=np.std(mean_set)#Store 1sigma error
        
        f=h5py.File('/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/dotproduct/spin_lss/DTFE_grid%d_spin_store_fil_Log%s-%s_smth%sMpc_%sbins.h5'%(grid_nodes,round(low_int_mass,2),round(hi_int_mass,2),smooth_scl,tot_mass_bins),'w')     
        f.create_dataset('/dp',data=costheta)
        f.close() 
    
#Plot correlation    
alignment_plt(grid_nodes,results,smooth_scl)
f=h5py.File('results_DTFE_grid%d_spin_store_fil_Log%s-%s_smth%sMpc_%sbins.h5'%(grid_nodes,round(low_int_mass,2),round(hi_int_mass,2),smooth_scl,tot_mass_bins),'w')     
f.create_dataset('/results',data=results)
f.close()


   
