import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sympy import *
import math as mth
import scipy
from scipy import fftpack
np.random.seed(1)
grid_nodes=30
#Create density field
image=np.zeros([grid_nodes,grid_nodes,grid_nodes])
img_x,img_y,img_z=np.shape(image)

for i in range(100):
    image[np.random.randint(0,grid_nodes),np.random.randint(0,grid_nodes),np.random.randint(0,grid_nodes)]=20

#image[15,15,15]=1 
'''
#Toggle the next two lines to either implement my_den or DTFE density fields
mu=0
sigma=1
z=10
np.random.seed(1)
qty_filament=400
qty_cluster=400
qty_points=qty_filament+qty_cluster
#Gernerate Filament
x=np.random.normal(mu,sigma,size=qty_filament)+75
y=np.random.normal(mu,sigma,size=qty_filament)+75
z=np.random.rand(qty_filament)*50+60
points=np.column_stack((x.flatten(),y.flatten(),z.flatten()))
#Generate clusters
v1=np.zeros((qty_cluster,3))
theta=2*np.pi*np.random.uniform(0,1,[qty_cluster,1])
phi=np.arccos(2*(np.random.uniform(0,1,[qty_cluster,1]))-1)
v1[:,0]=np.multiply(np.sin(phi[:,0]),np.cos(theta[:,0]))
v1[:,1]=np.multiply(np.sin(phi[:,0]),np.sin(theta[:,0]))
v1[:,2]=np.cos(phi[:,0])
mu=0
sigma=1
amp=np.random.normal(mu,sigma,size=qty_cluster)
amp=np.reshape(amp,(qty_cluster,1))
v1=np.multiply(amp,v1)
v1[:,0]=v1[:,0]+75
v1[:,1]=v1[:,1]+75
v1[:,2]=v1[:,2]+55
v2=np.zeros((qty_cluster,3))
v2[:,0]=v1[:,0]
v2[:,1]=v1[:,1]
v2[:,2]=v1[:,2]+60

box_sz=150.0
#points=np.vstack((points,v1,v2,np.array([[0,0,0],[box_sz,0,0],[0,box_sz,0],[0,0,box_sz],[box_sz,box_sz,0],[0,box_sz,box_sz],[box_sz,0,box_sz],[box_sz,box_sz,box_sz]])))
points=np.vstack((points,v1,v2))

#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(points[:,0],points[:,1],points[:,2])
#ax.view_init(elev=0., azim=-180)

Xc=np.asarray(points[:,0])
Yc=np.asarray(points[:,1])
Zc=np.asarray(points[:,2])

#Xc_min=np.min(points[:,0])
#Xc_max=np.max(points[:,0])
#Yc_min=np.min(points[:,1])
#Yc_max=np.max(points[:,1])
#Zc_min=np.min(points[:,2])
#Zc_max=np.max(points[:,2])

Xc_min=0.0
Xc_max=box_sz
Yc_min=0.0
Yc_max=box_sz
Zc_min=0.0
Zc_max=box_sz

grid_nodes=60

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001

grid=np.random.rand(grid_nodes,grid_nodes,grid_nodes)
#grid=np.zeros((grid_nodes,grid_nodes,grid_nodes))
halo_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
for i in range(len(Xc)):
   
    grid_index_x=mth.trunc(Xc[i]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(Yc[i]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(Zc[i]*Zc_mult-Zc_minus) 
    grid[grid_index_x,grid_index_y,grid_index_z]=grid[grid_index_x,grid_index_y,grid_index_z]+10
    halo_z[grid_index_x,grid_index_y,grid_index_z]+=10
image=halo_z
#image=ndimage.filters.gaussian_filter(image,4)

img_x,img_y,img_z=np.shape(image)
'''
s=4.0#standard deviation of gaussian
#
img = ndimage.gaussian_filter(image,s,order=0,mode='wrap',truncate=20)#smoothing function

#3d smoothing via convolution method
in_val,fnl_val=-15.0,15.0#kernel values
kernel_sz=grid_nodes
X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,kernel_sz),np.linspace(in_val,fnl_val,kernel_sz),np.linspace(in_val,fnl_val,kernel_sz))
h=(1/np.sqrt(1.0*2*np.pi*s*s))**(3)*np.exp(-1/(1.0*2*s*s)*(Y**2+X**2+Z**2))
'''
#Kernel padding
k_xx=np.zeros([img_x,img_y,img_z])
kern_in=1.*(len(k_xx)-len(h))/2
kern_fnl=1.*(len(k_xx)-len(h))/2+len(h)
k_xx[kern_in:kern_fnl,kern_in:kern_fnl,kern_in:kern_fnl]=h
'''
h=np.roll(h,int(grid_nodes/2),axis=0)
h=np.roll(h,int(grid_nodes/2),axis=1)
h=np.roll(h,int(grid_nodes/2),axis=2)
fft_dxx=scipy.fftpack.fftn(h)
fft_db=scipy.fftpack.fftn(image)
ifft_dxx=scipy.fftpack.ifftn(np.multiply(fft_dxx,fft_db)).real
ifft_dxx=np.clip(ifft_dxx,0,1)
#ifft_dxx=ndimage.convolve(h,image)

#calculate difference between smoothed fields
den_diff=abs(abs(ifft_dxx)-abs(img))
#den_diff_val_convfnc=sum(abs(ifft_dxx.flatten()-ifft_dxx_fnct.flatten()))

#plotting
plt.figure(figsize=(15,17))
#Plotting global features
slc=1
rows=['Min','Max','Total']
columns=['Density Values']
subtitl_offset=1.15
ttl_fnt=12
plt.subplots_adjust(hspace=0.3,wspace=0.12,top=0.85)
plt.suptitle("Density fld smoothing comparison.\nSmoothing scale: %s pxls \nSlice: %s/%s"%(s,slc,grid_nodes),y=0.95,fontsize=20)

ax1=plt.subplot2grid((2,2), (1,1))    
plt.title('ndimage function',y=subtitl_offset,fontsize=ttl_fnt)
cmapp = plt.get_cmap('jet')
scl_plt=1#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax1.imshow(np.power(img[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,30,0,30])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)
#generate sub-table
data_fnc=np.round(np.array([[np.min(img)],[np.max(img)],[sum(img.flatten())]]),8)
tbl_fnc=plt.table(cellText=data_fnc,loc='top',rowLabels=rows,colLabels=columns,colWidths=[0.5 for x in columns],cellLoc='center')
tbl_fnc.set_fontsize(12)
tbl_fnc.scale(1.2, 1.2)

ax2=plt.subplot2grid((2,2), (1,0))    
plt.title('my function',y=subtitl_offset,fontsize=ttl_fnt)
cmapp = plt.get_cmap('jet')
scl_plt=1#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax2.imshow(np.power(ifft_dxx[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,30,0,30])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)
#generate sub-table
data_fnc=np.round(np.array([[np.min(ifft_dxx)],[np.max(ifft_dxx)],[sum(ifft_dxx.flatten())]]),8)
tbl_my=plt.table(cellText=data_fnc,loc='top',rowLabels=rows,colLabels=columns,colWidths=[0.5 for x in columns],cellLoc='center')
tbl_my.set_fontsize(12)
tbl_my.scale(1.2, 1.2)

ax3=plt.subplot2grid((2,2), (0,0))    
plt.title('raw density field',y=subtitl_offset,fontsize=ttl_fnt)
cmapp = plt.get_cmap('jet')
scl_plt=1#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax3.imshow(np.power(image[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,30,0,30])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)
#generate sub-table
data_fnc=np.round(np.array([[np.min(image)],[np.max(image)],[sum(image.flatten())]]),8)
tbl_raw=plt.table(cellText=data_fnc,loc='top',rowLabels=rows,colLabels=columns,colWidths=[0.5 for x in columns],cellLoc='center')
tbl_raw.set_fontsize(12)
tbl_raw.scale(1.2, 1.2)

ax4=plt.subplot2grid((2,2), (0,1))    
plt.title('Difference b/w smthd fields',y=subtitl_offset,fontsize=ttl_fnt)
cmapp = plt.get_cmap('jet')
scl_plt=1#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax4.imshow(np.power(den_diff[slc,:,:],1.0/scl_plt),cmap=cmapp,extent=[0,30,0,30])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)
#generate sub-table
data_fnc=np.round(np.array([[np.min(den_diff)],[np.max(den_diff)],[sum(den_diff.flatten())]]),8)
tbl_diff=plt.table(cellText=data_fnc,loc='top',rowLabels=rows,colLabels=columns,colWidths=[0.5 for x in columns],cellLoc='center')
tbl_diff.set_fontsize(12)
tbl_diff.scale(1.2, 1.2)

#plt.savefig('/import/oth3/ajib0457/3d_smth_test/dot_box/smoothed_den_field_test_%spixels_bound%s_grid%s_slc%s_dot.png'%(s,fnl_val,grid_nodes,slc))
