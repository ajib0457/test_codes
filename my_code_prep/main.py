import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math as mth
import h5py
import sklearn.preprocessing as skl

sim_sz=60         #Mpc/h
grid_nodes=200  #Density grid resolution
smooth_scl=3.5    #Mpc/h
#Symbolize variables and specify function

f=h5py.File("/import/oth3/ajib0457/Peng_test_data_run/cat_reform/bolchoi_DTFE_rockstar_box_%scubed_xyz_vxyz_jxyz_m_r.h5"%sim_sz, 'r')#xyz vxvyvz jxjyjz & Rmass & Rvir: Halo radius (kpc/h comoving).
data=f['/halo'][:]
f.close()

Xc=data[:,0]
Yc=data[:,1]
Zc=data[:,2]
h_mass=data[:,9]
 
halos=np.column_stack((Xc,Yc,Zc))

#pre-binning for Halos ----------
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
#--------------------------------

#grid=np.zeros((grid_nodes,grid_nodes,grid_nodes))
image=np.zeros((grid_nodes,grid_nodes,grid_nodes))
for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus)   
    image[grid_index_x,grid_index_y,grid_index_z]+=h_mass[i] 

img_x,img_y,img_z=np.shape(image)

X,Y,Z,s=symbols('X Y Z s')
h=(1/sqrt(2*pi*s*s))**(3)*exp(-1/(2*s*s)*(Y**2+X**2+Z**2))
#take second partial derivatives as per Hessian 
hprimexx=h.diff(X,X)
hprimexy=h.diff(X,Y)
hprimeyy=h.diff(Y,Y)
hprimezz=h.diff(Z,Z)
hprimezx=h.diff(Z,X)
hprimezy=h.diff(Z,Y)
#Lambdify i.e make them variables once again
fxx=lambdify((X,Y,Z,s),hprimexx,'numpy')
fxy=lambdify((X,Y,Z,s),hprimexy,'numpy')
fyy=lambdify((X,Y,Z,s),hprimeyy,'numpy')
fzz=lambdify((X,Y,Z,s),hprimezz,'numpy')
fzx=lambdify((X,Y,Z,s),hprimezx,'numpy')
fzy=lambdify((X,Y,Z,s),hprimezy,'numpy')
#3d meshgrid for each kernel and evaluate 6 partial derivatives to generate 9 kernels

#Kernal settings
s=1.0*smooth_scl/sim_sz*grid_nodes
kern_x,kern_y,kern_z=img_x,img_y,img_z #kernel size
in_val,fnl_val=-grid_nodes/2,grid_nodes/2

#Kernel generator
X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,kern_y),np.linspace(in_val,fnl_val,kern_x),np.linspace(in_val,fnl_val,kern_z))
dxx=fxx(X,Y,Z,s)
dxy=fxy(X,Y,Z,s)
dyy=fyy(X,Y,Z,s)
dzz=fzz(X,Y,Z,s)
dzx=fzx(X,Y,Z,s)
dzy=fzy(X,Y,Z,s)
   
#shift kernel to 0,0
dxx=np.roll(dxx,int(img_x/2),axis=0)
dxx=np.roll(dxx,int(img_y/2),axis=1)
dxx=np.roll(dxx,int(img_z/2),axis=2)
dxy=np.roll(dxy,int(img_x/2),axis=0)
dxy=np.roll(dxy,int(img_y/2),axis=1)
dxy=np.roll(dxy,int(img_z/2),axis=2)
dyy=np.roll(dyy,int(img_x/2),axis=0)
dyy=np.roll(dyy,int(img_y/2),axis=1)
dyy=np.roll(dyy,int(img_z/2),axis=2)
dzz=np.roll(dzz,int(img_x/2),axis=0)
dzz=np.roll(dzz,int(img_y/2),axis=1)
dzz=np.roll(dzz,int(img_z/2),axis=2)
dzx=np.roll(dzx,int(img_x/2),axis=0)
dzx=np.roll(dzx,int(img_y/2),axis=1)
dzx=np.roll(dzx,int(img_z/2),axis=2)
dzy=np.roll(dzy,int(img_x/2),axis=0)
dzy=np.roll(dzy,int(img_y/2),axis=1)
dzy=np.roll(dzy,int(img_z/2),axis=2)
#fft 6 kernels
fft_dxx=np.fft.fftn(dxx)
fft_dxy=np.fft.fftn(dxy)
fft_dyy=np.fft.fftn(dyy)
fft_dzz=np.fft.fftn(dzz)
fft_dzx=np.fft.fftn(dzx)
fft_dzy=np.fft.fftn(dzy)
fft_db=np.fft.fftn(image)
#convolution of kernels with density field & inverse transform
ifft_dxx=np.fft.ifftn(np.multiply(fft_dxx,fft_db)).real
ifft_dxy=np.fft.ifftn(np.multiply(fft_dxy,fft_db)).real
ifft_dyy=np.fft.ifftn(np.multiply(fft_dyy,fft_db)).real
ifft_dzz=np.fft.ifftn(np.multiply(fft_dzz,fft_db)).real
ifft_dzx=np.fft.ifftn(np.multiply(fft_dzx,fft_db)).real
ifft_dzy=np.fft.ifftn(np.multiply(fft_dzy,fft_db)).real
#reshape into column matrices
ifft_dxx=np.reshape(ifft_dxx,(np.size(ifft_dxx),1))      
ifft_dxy=np.reshape(ifft_dxy,(np.size(ifft_dxy),1))
ifft_dyy=np.reshape(ifft_dyy,(np.size(ifft_dyy),1))
ifft_dzz=np.reshape(ifft_dzz,(np.size(ifft_dzz),1))
ifft_dzx=np.reshape(ifft_dzx,(np.size(ifft_dzx),1))
ifft_dzy=np.reshape(ifft_dzy,(np.size(ifft_dzy),1))
#create Hessian
hessian=np.column_stack((ifft_dxx,ifft_dxy,ifft_dzx,ifft_dxy,ifft_dyy,ifft_dzy,ifft_dzx,ifft_dzy,ifft_dzz))
   
del ifft_dxx
del ifft_dxy
del ifft_dyy
del ifft_dzz
del ifft_dzx
del ifft_dzy

hessian=np.reshape(hessian,(grid_nodes**3,3,3))#change to 3,3 for 3d and 1,1 for 1d
#calculate eigenvalues and eigenvectors
eig_vals_vecs=np.linalg.eig(hessian)
del hessian

#extract eigenvalues
eigvals_unsorted=eig_vals_vecs[0]
eigvals=np.sort(eigvals_unsorted)
#extract eigenvectors
eigvecs=eig_vals_vecs[1]
eig_one=eigvals[:,2]
eig_two=eigvals[:,1]
eig_three=eigvals[:,0]

#link eigenvalues as keys to eigenvectors as values inside dictionary    
vec_arr_num,vec_row,vec_col=np.shape(eigvecs)
values=np.reshape(eigvecs.transpose(0,2,1),(vec_row*vec_arr_num,vec_col))#orient eigenvectors so that each row is an eigenvector

eigvals_unsorted=eigvals_unsorted.flatten()
vecsvals=np.column_stack((eigvals_unsorted,values))

lss=['filament','sheet','void','cluster']#Choose which LSS you would like to get classified
def lss_classifier(lss,eigvals_unsorted,values,eig_one,eig_two,eig_three):
    
    ####Classifier#### 
    '''
    This is the classifier function which takes input:
    
    lss: the labels of Large scale structure which will be identified pixel by pixel and also eigenvectors 
    will be retrieved if applicable.
    
    vecsvals: These are the eigenvalues and eigevector pairs which correspond row by row.

    eig_one,two and three: These are the isolated eigenvalues 
    
    This function will output:
    
    eig_fnl: An array containing all of the relevent eigenvectors for each LSS type
    
    mask_fnl: array prescribing 0-void, 1-sheet, 2-filament and 3-cluster
    '''
    eig_fnl=np.zeros((grid_nodes**3,4))
    mask_fnl=np.zeros((grid_nodes**3))
    for i in lss:
        vecsvals=np.column_stack((eigvals_unsorted,values))
        recon_img=np.zeros([grid_nodes**3])
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
        
        #LSS FILTER#
        
        recon_img[recon_filt_one]=1
        recon_img[recon_filt_two]=recon_img[recon_filt_two]+1
        recon_img[recon_filt_three]=recon_img[recon_filt_three]+1  
        del recon_filt_one
        del recon_filt_two
        del recon_filt_three
        recon_img=recon_img.flatten()
        recon_img=recon_img.astype(np.int8)
        mask=(recon_img !=3)#Up to this point, a mask is created to identify where there are NO filaments...
        mask_true=(recon_img ==3)
        del recon_img
        vecsvals=np.reshape(vecsvals,(grid_nodes**3,3,4))
        
        
        #Find relevent eigpairs
        if (i=='void'):#There is no appropriate axis of a void?
            mask_fnl[mask_true]=0
            del mask_true
            
        if (i=='sheet'):
            vecsvals[mask,:,:]=np.ones((3,4))*9#...which are then converted into -9 at this point
            del mask
            fnd_prs=np.where(vecsvals[:,:,0]<0)#find LSS axis
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]
            mask_fnl[mask_true]=1
            del mask_true

        if (i=='filament'):
            vecsvals[mask,:,:]=np.ones((3,4))*-9#...which are then converted into -9 at this point
            del mask
            fnd_prs=np.where(vecsvals[:,:,0]>=0)#find LSS axis
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]
            mask_fnl[mask_true]=2
            del mask_true
            
        if (i=='cluster'):#There is no appropriate axis of a void?
            mask_fnl[mask_true]=3
            del mask_true            
        
    return eig_fnl,mask_fnl  
    
eig_fnl,mask_fnl= lss_classifier(lss,eigvals_unsorted,values,eig_one,eig_two,eig_three)#Function run
del eig_three
del eig_two
del eig_one  
del values  

recon_vecs_x=eig_fnl[:,1]
recon_vecs_y=eig_fnl[:,2]
recon_vecs_z=eig_fnl[:,3]

#Dot Product
recon_vecs_flt_unnorm=np.column_stack((recon_vecs_x,recon_vecs_y,recon_vecs_z))
del recon_vecs_x
del recon_vecs_y
del recon_vecs_z
mask=np.reshape(mask_fnl,(grid_nodes,grid_nodes,grid_nodes))

recon_vecs_flt_norm=skl.normalize(recon_vecs_flt_unnorm)#I should not normalize becauase they are already normalized and also the classifier (9) mask will be ruined
recon_vecs=np.reshape(recon_vecs_flt_norm,(grid_nodes,grid_nodes,grid_nodes,3))#Three for the 3 vector components
del recon_vecs_flt_norm
recon_vecs_unnorm=np.reshape(recon_vecs_flt_unnorm,(grid_nodes,grid_nodes,grid_nodes,3))#raw eigenvectors along with (9)-filled rows which represent blank vectors
del recon_vecs_flt_unnorm
# -----------------

tot_mass_bins=4

partcl_500=np.where((data[:,9]/(1.35*10**8))>=500)#filter out halos with <500 particles
data=data[partcl_500]
halo_mass=data[:,9]
log_halo_mass=np.log10(halo_mass)#convert into log(M)
mass_intvl=(np.max(log_halo_mass)-np.min(log_halo_mass))/tot_mass_bins


for mass_bin in range(tot_mass_bins):
    
    low_int_mass=np.min(log_halo_mass)+mass_intvl*mass_bin
    hi_int_mass=low_int_mass+mass_intvl
    mass_mask=np.zeros(len(log_halo_mass))
    loint=np.where(log_halo_mass>=low_int_mass)#Change these two numbers as according to the above intervals
    hiint=np.where(log_halo_mass<hi_int_mass)#Change these two numbers as according to the above intervals
    mass_mask[loint]=1
    mass_mask[hiint]=mass_mask[hiint]+1
    mass_indx=np.where(mass_mask==2)
    
    #Angular momentum
    Lx=data[:,6]
    Ly=data[:,7]
    Lz=data[:,8]
    #Positions
    Xc=data[:,0]
    Xc=Xc[mass_indx]
    Yc=data[:,1]
    Yc=Yc[mass_indx]
    Zc=data[:,2]
    Zc=Zc[mass_indx]
    
    #normalized angular momentum vectors v1
    halos_mom=np.column_stack((Lx,Ly,Lz))
    halos_mom=halos_mom[mass_indx]
    norm_halos_mom=skl.normalize(halos_mom)
    halos=np.column_stack((Xc,Yc,Zc,norm_halos_mom))
    # -----------------
    
    #grid=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    store_spin=[]
    for i in range(len(Xc)):
       #Create index related to the eigenvector bins
        grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
        grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
        grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus) 
        #calculate dot product and bin
        if (mask[grid_index_x,grid_index_y,grid_index_z]==2):#condition includes recon_vecs_unnorm so that I may normalize the vectors which are being processed
            spin_dot=np.inner(halos[i,3:6],recon_vecs[grid_index_x,grid_index_y,grid_index_z,:]) 
            store_spin.append(spin_dot)
    
    del halos
    store_spin=np.asarray(store_spin) 
    costheta=abs(store_spin)
    #in terms of pixels
    pxl_length_gauss=1.*s/(2*fnl_val)*grid_nodes
    #in terms of Mpc/h
    std_dev_phys=pxl_length_gauss*(1.*sim_sz/grid_nodes)
          
    #Correlation 
    mean_val=np.mean(costheta)
    #bootstrap resampling error
    runs=200+mass_bin*300
    a=np.random.randint(low=0,high=len(costheta),size=(runs,len(costheta)))
    mean_set=np.mean(costheta[a],axis=1)
    del a
    del costheta

    print('MLE=%s +-%s'%(round(mean_val,4),round(np.std(mean_set),4)))
    
    
