import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math as mth
import h5py
import sklearn.preprocessing as skl
from plotter_funcs import * 
from scipy import ndimage

sim_sz=60           #Size of simulation in physical units Mpc/h cubed
grid_nodes=200      #Density Field grid resolution
smooth_scl=2      #Smoothing scale in physical units Mpc/h
tot_mass_bins=4     #Number of Halo mass bins
particles_filt=300  #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
Mass_res=1.35*10**8 #Bolchoi particle mass as per, https://arxiv.org/pdf/1002.3660.pdf

#Load Bolchoi Simulation Catalogue, ONLY filtered for X,Y,Z=<sim_sz
f=h5py.File("bolchoi_DTFE_rockstar_box_%scubed_xyz_vxyz_jxyz_m_r.h5"%sim_sz, 'r')
#f=h5py.File("/import/oth3/ajib0457/wang_peng_code/Peng_run_bolchoi_smpl_python/bolchoi_DTFE_rockstar_allhalos_xyz_vxyz_jxyz_m_r.h5", 'r')
data=f['/halo'][:]#data array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(Mpc/h)*km/s), (Vir. Mass)Mvir(Msun/h) & (Vir. Rad)Rvir(kpc/h) 
f.close()

Xc=data[:,0]#halo X coordinates
Yc=data[:,1]#halo Y coordinates
Zc=data[:,2]#halo Z coordinates
h_mass=data[:,9]#halo Virial Mass
halos=np.column_stack((Xc,Yc,Zc))

# SECTION 1. Density field creation ------------------------------------------------
#Manual technique to bin halos within a 3D Matrix
Xc_min=np.min(Xc)
Xc_max=np.max(Xc)
Yc_min=np.min(Yc)
Yc_max=np.max(Yc)
Zc_min=np.min(Zc)
Zc_max=np.max(Zc)

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001

image=np.zeros((grid_nodes,grid_nodes,grid_nodes))
for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus)   
    image[grid_index_x,grid_index_y,grid_index_z]+=h_mass[i]#Add halo mass to coinciding pixel 
#END SECTION-------------------------------------------------------------------------
      
# **IGNORE** SECTION 1.1 Plot scatter, velocity & AM vector scatter----------------------------
slc=160             # 0 - grid_nodes  
plane=1             #X-0, Y-1, Z-2   
plane_thickness=5 #Mpc/h
scatter_plot(Xc,Yc,Zc,slc,plane,grid_nodes,plane_thickness,sim_sz)
#-----------------------------------------------------------------------------------

# SECTION 2. Create Hessian ---------------------------------------------------------------------------------------
#smooth via ndimage function
s=1.0*smooth_scl/sim_sz*grid_nodes# s- standard deviation of Kernel, converted from Mpc/h into number of pixels
smthd_image = ndimage.gaussian_filter(image,s,order=0,mode='wrap',truncate=25)#smoothing function
fft_smthd_image=np.fft.fftn(smthd_image)

#Create k-space grid
k_grid = range(grid_nodes)/np.float64(grid_nodes)
for i in range(grid_nodes/2+1, grid_nodes):
    k_grid[i] = -np.float64(grid_nodes-i)/np.float64(grid_nodes)

rc=1.*sim_sz/(grid_nodes)#physical space interval
k_grid = k_grid*2*np.pi/rc# k=2pi/lambda as per the definition of a wavenumber. see wiki

k_z=np.reshape(k_grid,(1,grid_nodes))
k_y=np.reshape(k_z,(grid_nodes,1))

a_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
a_z[:]=k_z
a_y=np.zeros((grid_nodes,grid_nodes,grid_nodes))
a_y[:]=k_y
a_x=np.zeros((grid_nodes,grid_nodes,grid_nodes))
a_x=a_z.transpose()

a_xx=-1*np.multiply(a_x,a_x)
a_yy=-1*np.multiply(a_y,a_y)
a_zz=-1*np.multiply(a_z,a_z)
a_xy=-1*np.multiply(a_x,a_y)
a_xz=-1*np.multiply(a_x,a_z)
a_yz=-1*np.multiply(a_y,a_z)
del a_z
del a_y
del a_x
dxx=fft_smthd_image*a_xx
dyy=fft_smthd_image*a_yy
dzz=fft_smthd_image*a_zz
dxy=fft_smthd_image*a_xy
dxz=fft_smthd_image*a_xz
dyz=fft_smthd_image*a_yz
del a_xx
del a_yy
del a_zz
del a_xy
del a_xz
del a_yz
dxx=np.fft.ifftn(dxx).real
dxx=dxx.flatten()
dyy=np.fft.ifftn(dyy).real
dyy=dyy.flatten()
dzz=np.fft.ifftn(dzz).real
dzz=dzz.flatten()
dxy=np.fft.ifftn(dxy).real
dxy=dxy.flatten()
dxz=np.fft.ifftn(dxz).real
dxz=dxz.flatten()
dyz=np.fft.ifftn(dyz).real
dyz=dyz.flatten()

hessian=np.column_stack((dxx,dxy,dxz,dxy,dyy,dyz,dxz,dyz,dzz))
del dxx
del dyy
del dzz
del dxy
del dxz
del dyz
hessian=np.reshape(hessian,(grid_nodes**3,3,3))

f=h5py.File("my_hess.h5", 'w')
f.create_dataset('Hessian_matrix', data=hessian)
f.close()
#END SECTION----------------------------------------------------------------------------------------------------

'''
# **IGNORE** SECTION 2.1 Plot scatter,smoothed, eigvals, classification mask,color scatter, table with halo-classif ratios----------------------------
scl_plt=50 #nth root scaling of density field
smoothed_density_field(grid_nodes,slc,smooth_scl,in_val,fnl_val,s,image,plane,scl_plt)
#-----------------------------------------------------------------------------------
'''
# SECTION 3. Calculate and reorder eigenpairs----------------------------------------------
eig_vals_vecs=np.linalg.eig(hessian)
#Eigenvalues 
eigvals_unsorted=eig_vals_vecs[0]
eigvals=np.sort(eigvals_unsorted)
eig_one=eigvals[:,2]
eig_two=eigvals[:,1]
eig_three=eigvals[:,0]
eigvals_unsorted=eigvals_unsorted.flatten()
#Eigenvectors
eigvecs=eig_vals_vecs[1]
vec_arr_num,vec_row,vec_col=np.shape(eigvecs)
values=np.reshape(eigvecs.transpose(0,2,1),(vec_row*vec_arr_num,vec_col))#orient eigenvectors so that each row is an eigenvector
#END SECTION------------------------------------------------------------------------------
'''
#SECTION 3.1 plot eigenvalues-------------------------------------------
eigenvalue_plts(eig_one,eig_two,eig_three,grid_nodes,sim_sz,smooth_scl)
#------------------------------------------------------------------------
'''

# SECTION 4. Classify Large scale structure--------------------------------------------------------------------
lss=['filament','sheet','void','cluster']#which LSS to output mask & eigenpairs for
def lss_classifier(lss,eigvals_unsorted,values,eig_one,eig_two,eig_three):   
    '''
    This is the classifier function which takes input:
    
    lss: the labels of Large scale structure which will be identified pixel by pixel and also eigenvectors 
    will be retrieved if applicable.
    
    eigvals_unsorted: flattened eigenvalues of entire simulation
    
    values: eigenvectors of entire simulation reshaped into row vectors

    eig_one,two and three: These are the isolated eigenvalues 
    
    This function will output:
    
    eig_fnl: An array containing all of the relevent eigenvectors for each LSS type
    
    mask_fnl: array prescribing 0-void, 1-sheet, 2-filament and 3-cluster
    '''
    eig_fnl=np.zeros((grid_nodes**3,4))#first column will store all relevent eigenvalues and last 3 columns will store eigenvectors
    mask_fnl=np.zeros((grid_nodes**3))
    for i in lss:#Will loop once for each LSS chosen using 'lss' list
        vecsvals=np.column_stack((eigvals_unsorted,values))#Redefine vecsvals for each iteration
        vecsvals=np.reshape(vecsvals,(grid_nodes**3,3,4))#Each pixel is 3x4 where first column is eigenvalues and next 3 col are eigenvectors
        recon_img=np.zeros([grid_nodes**3])#Mask used along with below conditions to mask out pixels 
        #Conditions for each LSS
        if (i=='void'):
            recon_filt_one=np.where(eig_three>0)
            recon_filt_two=np.where(eig_two>0)
            recon_filt_three=np.where(eig_one>0)
        if (i=='sheet'):
            recon_filt_one=np.where(eig_three<0)
            recon_filt_two=np.where(eig_two>=0)
            recon_filt_three=np.where(eig_one>=0)
        if (i=='filament'):
            recon_filt_one=np.where(eig_three<0)
            recon_filt_two=np.where(eig_two<0)
            recon_filt_three=np.where(eig_one>=0)
        if (i=='cluster'):
            recon_filt_one=np.where(eig_three<0)
            recon_filt_two=np.where(eig_two<0)
            recon_filt_three=np.where(eig_one<0)
                
        recon_img[recon_filt_one]=1
        recon_img[recon_filt_two]=recon_img[recon_filt_two]+1
        recon_img[recon_filt_three]=recon_img[recon_filt_three]+1  

        recon_img=recon_img.flatten()
        recon_img=recon_img.astype(np.int8)#Just done to reduce memory
        mask=(recon_img !=3)
        mask_true=(recon_img ==3)
        del recon_img
                
        #Find and store relevent eigpairs
        if (i=='void'):
            mask_fnl[mask_true]=0#Identify final mask void-0
            
        if (i=='sheet'):
            vecsvals[mask,:,:]=np.ones((3,4))*9#wipe out all pixels which are not sheet

            fnd_prs=np.where(vecsvals[:,:,0]<0)#Find where eigenvectors which are perpendicular to sheet plane
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]#save relevent eigpairs into final output array
            mask_fnl[mask_true]=1#Identify final mask sheet-1

        if (i=='filament'):
            vecsvals[mask,:,:]=np.ones((3,4))*-9#wipe out all pixels which are not filament

            fnd_prs=np.where(vecsvals[:,:,0]>=0)#Find e3 eigenvectors using eigenvalues
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]#save relevent eigpairs into final output array
            mask_fnl[mask_true]=2#Identify final mask filament-2
            
        if (i=='cluster'):
            mask_fnl[mask_true]=3#Identify final mask cluster-3
            
    return eig_fnl,mask_fnl  
eig_fnl,mask_fnl= lss_classifier(lss,eigvals_unsorted,values,eig_one,eig_two,eig_three)#Function run
mask=np.reshape(mask_fnl,(grid_nodes,grid_nodes,grid_nodes))#Reshape mask_fnl
#END SECTION-----------------------------------------------------------------------------------------------------

f=h5py.File("eig_fnl_mask.h5", 'w')
f.create_dataset('eig_fnl', data=eig_fnl)
f.create_dataset('mask_fnl', data=mask_fnl)
f.close()

# **IGNORE** SECTION 4.1 Plot scatter,smoothed, eigvals, classification mask,color scatter, table with halo-classif ratios----------------------------
classify_mask(mask,grid_nodes,slc,smooth_scl,plane)
colorscatter_plt(data,mask,grid_nodes,slc,smooth_scl,plane,sim_sz,plane_thickness,Mass_res,particles_filt,Xc_mult,Xc_minus,Yc_mult,Yc_minus,Zc_mult,Zc_minus)
#-----------------------------------------------------------------------------------

# SECTION 5. Take Dot Product of Spin-LSS------------------------------------------------------------------------
recon_vecs_flt_unnorm=eig_fnl[:,1:4]#take eigenvectors from eig_fnl
recon_vecs_flt_norm=skl.normalize(recon_vecs_flt_unnorm)#normalize eigenvectors to make sure.
recon_vecs=np.reshape(recon_vecs_flt_norm,(grid_nodes,grid_nodes,grid_nodes,3))#Reshape eigenvectors

#'data' format reminder: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(Mpc/h)*km/s), (Vir. Mass)Mvir(Msun/h) & (Vir. Rad)Rvir(kpc/h)
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
          
    results[mass_bin,2]=np.mean(costheta)#Alignment value calc. and store
    
    #Calculating error using bootstrap resampling
    runs=200+mass_bin*300
    a=np.random.randint(low=0,high=len(costheta),size=(runs,len(costheta)))
    mean_set=np.mean(costheta[a],axis=1)
    results[mass_bin,3]=np.std(mean_set)#Store 1sigma error
    
print(results)    
#END SECTION & CODE---------------------------------------------------------------------------------------------------

# **IGNORE** SECTION 5.1 Plot correlation----------------------------
alignment_plt(grid_nodes,results)
vector_scatter(data,mask,recon_vecs,grid_nodes,sim_sz,particles_filt,slc,Xc_mult,Xc_minus,Yc_mult,Yc_minus,Zc_mult,Zc_minus,plane_thickness)
#-----------------------------------------------------------------------------------
