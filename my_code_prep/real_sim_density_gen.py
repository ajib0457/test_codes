import numpy as np
import math as mth
import pynbody
import h5py
import sys

sim_type=sys.argv[1]           #'dm_only' 'dm_gas'
cosmology=sys.argv[2]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot=int(sys.argv[3])      #'12  '11'...
den_type='my_den'              #not an option for this code
grid_nodes=1250                #density resolution

#Assigns v to angular momentum of halos
f=pynbody.load("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/snapshot_0%s"%(sim_type,cosmology,snapshot,snapshot))
a=f.dm['pos']
dm_mass=np.unique(f.dm['mass'])
if sim_type=='dm_gas': 
    a=np.vstack((a,f.gas['pos']))
    gas_mass=np.unique(f.gas['mass'])

Xc=np.asarray(a[:,0]).astype(float)
Yc=np.asarray(a[:,1]).astype(float)
Zc=np.asarray(a[:,2]).astype(float)

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

grid=np.zeros((grid_nodes,grid_nodes,grid_nodes))

for i in range(len(f.dm['pos'])):
   
    grid_index_x=mth.trunc(Xc[i]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(Yc[i]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(Zc[i]*Zc_mult-Zc_minus)     
    grid[grid_index_x,grid_index_y,grid_index_z]=grid[grid_index_x,grid_index_y,grid_index_z]+dm_mass

if sim_type=='dm_gas':
    for i in range(i+1,len(Xc)):
       
        grid_index_x=mth.trunc(Xc[i]*Xc_mult-Xc_minus)      
        grid_index_y=mth.trunc(Yc[i]*Yc_mult-Yc_minus) 
        grid_index_z=mth.trunc(Zc[i]*Zc_mult-Zc_minus) 
        grid[grid_index_x,grid_index_y,grid_index_z]=grid[grid_index_x,grid_index_y,grid_index_z]+gas_mass

grid=grid.flatten()

f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/dens/%s/pascal_%sgrid_interp_%s_%s_snapshot_0%s_gd%s.h5" %(sim_type,cosmology,snapshot,den_type,den_type,sim_type,cosmology,snapshot,grid_nodes), 'w')
f.create_dataset('/grid_int',data=grid)
f.close()
