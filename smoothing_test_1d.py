import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sympy import *
import h5py

grid_nodes=850

f1=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/DTFE_out_%d_bolchoi_rockstar_1dslc.a_den.h5" %(grid_nodes), 'r')
image=f1['/DTFE'][:]
f1.close()
img_x=len(image)

#my function
in_val,fnl_val=-grid_nodes/2,grid_nodes/2#kernel values
X,s=symbols('X s')#Needed for sympy to take derivative 
h=(1.0/sqrt(2.0*pi*s*s))**(1)*exp(-1/(1.0*2*s*s)*(X**2))
hprimexx=h.diff(X,X)#Take second derivative dxx
fxx=lambdify((X,s),hprimexx,'numpy')#Needed for sympy to take derivative

s=11.9# Standard deviation
X=np.linspace(in_val,fnl_val,img_x)
dxx=fxx(X,s)#Second derivative of Gaussian equation h
h=np.roll(dxx,int(img_x/2.0))
fft_kernal=np.fft.fft(h)
fft_image=np.fft.fft(image)
ifft_secndder=np.fft.ifft(np.multiply(fft_kernal,fft_image)).real

#ndimage function
img = ndimage.gaussian_filter(image,s,order=2,mode='wrap',truncate=30)

#subtraction
subt=sum(abs(img)-abs(ifft_secndder))

#plotting
plt.figure(figsize=(12,15))

ax1=plt.subplot2grid((2,1), (0,0))
plt.plot(image,label='Density Field' )
plt.legend(loc='left')
plt.suptitle("Method subtraction %s"%round(subt,4),y=0.95,fontsize=15)
plt.title('Density Field of %s pixels'%grid_nodes)
ax1=plt.subplot2grid((2,1), (1,0))
scl_plt=5
plt.plot(ifft_secndder,label='convolution method')
plt.plot(img,label='ndimage function')
plt.legend(loc='left')
plt.title('smoothed(sigma=%s pixels) second derivative of density field'%s)

plt.savefig('/import/oth3/ajib0457/3d_smth_test/1d/second_der_den_field_test_%spixels_bound%s_grid%s_1d_bolchoislc.png'%(s,fnl_val,grid_nodes))
