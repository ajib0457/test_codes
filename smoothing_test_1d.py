import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sympy import *
from astropy.convolution import Gaussian1DKernel, convolve
grid_nodes=850
#Create density field
image=np.zeros([grid_nodes])
img_x=len(image)
#image[5:10]=255
image[30:100]=2
image[180:300]=25
image[110:140]=25
image[310:340]=255
image[700:800]=255

#standard deviation of gaussian
s=9.0
#ndimage function
img = ndimage.gaussian_filter(image,s,order=0,mode='wrap',truncate=100)
#astropy function
g = Gaussian1DKernel(stddev=s,x_size=849)
z = convolve(image, g, boundary='wrap')

#my function
kern_x=img_x#kernel size
in_val,fnl_val=-425,425#kernel values
X=np.linspace(in_val,fnl_val,kern_x)
h=(1.0/np.sqrt(2.0*np.pi*s*s))**(1)*np.exp(-1/(1.0*2*s*s)*(X**2))
#padd=np.zeros([img_x])
#padd[int(img_x/2-kern_x/2):0+int(img_x/2+kern_x/2)]=h
h=np.roll(h,int(img_x/2.0))
fft_dxx=np.fft.fft(h)
fft_db=np.fft.fft(image)
ifft_dxx=np.fft.ifft(np.multiply(fft_dxx,fft_db)).real

#subtraction
print(abs(sum(z))-abs(sum(ifft_dxx)))

#plotting
plt.figure(figsize=(12,15))

ax1=plt.subplot2grid((2,1), (0,0))
plt.plot(ifft_dxx,label='convolution mthd')
plt.plot(img,label='ndimage fctn')
plt.plot(image,label='density fld')
plt.plot(z,label='astropy fctn')

plt.legend(loc='left')

ax1=plt.subplot2grid((2,1), (1,0))
scl_plt=20
plt.plot(np.power(ifft_dxx,1.0/scl_plt),label='convolution mthd')
plt.plot(np.power(img,1.0/scl_plt),label='ndimage fctn')
plt.plot(np.power(image,1.0/scl_plt),label='Density fld')
plt.plot(np.power(z,1.0/scl_plt),label='astropy fctn')

plt.savefig('/import/oth3/ajib0457/3d_smth_test/1d/smoothed_den_field_test_%spixels_bound%s_grid%s_1d.png'%(s,fnl_val,grid_nodes))
