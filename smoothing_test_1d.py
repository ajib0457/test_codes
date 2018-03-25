import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sympy import *
import h5py

sim_sz=250
grid_nodes=850
smooth_scl=2
f1=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/DTFE_out_%d_bolchoi_rockstar_1dslc.a_den.h5" %(grid_nodes), 'r')
image=f1['/DTFE'][:]
f1.close()
img_x=len(image)
s=1.0*smooth_scl/sim_sz*grid_nodes# s- standard deviation of Kernel, converted from Mpc/h into number of pixels
in_val,fnl_val=-grid_nodes/2,grid_nodes/2#kernel values
X,dxx=np.linspace(in_val,fnl_val,img_x,retstep=True)
#my function

#X,s=symbols('X s')#Needed for sympy to take derivative 
#h=(1.0/sqrt(2.0*pi*s*s))**(1)*exp(-1/(1.0*2*s*s)*(X**2))
h=(1/np.sqrt(1.0*2*np.pi*s*s))**(1)*np.exp(-1/(1.0*2*s*s)*((X-0.5)**2))
#hprimexx=h.diff(X,X)#Take second derivative dxx
#fxx=lambdify((X,s),hprimexx,'numpy')#Needed for sympy to take derivative
area=np.sum(h)
h=h/area


#dxx=fxx(X,s)#Second derivative of Gaussian equation h
h=np.roll(h,int(img_x/2.0))
fft_kernal=np.fft.fft(h)
fft_image=np.fft.fft(image)
ifft_secndder=np.fft.ifft(np.multiply(fft_kernal,fft_image)).real

#ndimage function
img = ndimage.gaussian_filter(image,s,order=0,mode='wrap',truncate=30)

#-----------------------
# Gaussian smooth PENG METHOD
#-----------------------
image_peng=np.fft.fftn(image)
rc=1.*sim_sz/(grid_nodes)
### creat k-space grid
kx = range(grid_nodes)/np.float64(grid_nodes)

for i in range(grid_nodes/2+1, grid_nodes):
    kx[i] = -np.float64(grid_nodes-i)/np.float64(grid_nodes)

kx = kx*2*np.pi/rc
ky = kx
kz = kx

kx2 = kx**2
ky2 = ky**2
kz2 = kz**2

# smooth_scl**2
Rs2 = np.float64(smooth_scl**2)/2.

print 'do smoothing in k-space will be much easier ....'
for i in range(grid_nodes):

    index = kx2[i]
    # smoothing in k-space
    image_peng[i] = np.exp(-index*Rs2)*image_peng[i]#convolving 1 pixel at a time

# transform to real space and save it
smoothed_image = np.fft.ifftn(image_peng).real
#---------------------------------

#subtraction
ndimag_mns_my=np.sum(abs(img)-abs(ifft_secndder))
ndimag_mns_peng=np.sum(abs(img)-abs(smoothed_image))


#plotting
plt.figure(figsize=(12,15))

ax1=plt.subplot2grid((2,1), (0,0))
plt.plot(image,label='Density Field' )
plt.legend(loc='left')
plt.suptitle("Method subtraction: ndimag_mns_my%s ndimag_mns_peng%s"%(ndimag_mns_my,ndimag_mns_peng),y=0.95,fontsize=15)
plt.title('Density Field of %s pixels'%grid_nodes)
ax1=plt.subplot2grid((2,1), (1,0))
scl_plt=5
plt.plot(ifft_secndder,label='convolution method')
plt.plot(img,label='ndimage function')
plt.plot(smoothed_image,label='Peng function')
plt.legend(loc='left')
plt.title('smoothed(sigma=%s pixels) density field'%s)


plt.savefig('/import/oth3/ajib0457/3d_smth_test/1d/smth_den_field_test_%spixels_bound%s_grid%s_1d_bolchoislc.png'%(s,fnl_val,grid_nodes))
