def scatter_plot(Xc,Yc,Zc,slc,plane,grid_nodes,plane_thickness,sim_sz):   
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    halos=np.column_stack((Xc,Yc,Zc))
            
    box=np.max(halos[:,0])#subset box length
    lo_lim_partcl=1.*slc/(grid_nodes)*box-1.*plane_thickness/2 #For particle distribution
    hi_lim_partcl=lo_lim_partcl+plane_thickness #For particle distributionn
    
    #Filter particles and halos
    partcls=np.array([0,0,0])
    
    for i in range(len(halos)):
       
        if (lo_lim_partcl<halos[i,plane]<hi_lim_partcl):#incremenets are in 0.00333 #wherver slc is, [x,y,z] make sure corresponds with if statement
            #density field
            result_hals=halos[i,:]
            partcls=np.row_stack((partcls,result_hals))
    partcls = np.delete(partcls, (0), axis=0)
    
    #Plotting
    fig, ax = plt.subplots(figsize=(14,14),dpi=100)
    ax.scatter(partcls[:,0],partcls[:,2],c='r')
    ax.set_xlim([0,sim_sz])
    ax.set_ylim([0,sim_sz]) 
    plt.xlabel('x[Mpc/h]') 
    plt.ylabel('y[Mpc/h]')
    plt.title('Scatter plot')
    ax.grid(True)
    plt.savefig('SCATTER_PLOT_gd%d_slc%d_thck%sMpc_%splane.png' %(grid_nodes,slc,plane_thickness,plane))
    
    return

def smoothed_density_field(grid_nodes,slc,smooth_scl,in_val,fnl_val,s,image,plane,scl_plt):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    kernel_sz=grid_nodes
    X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,kernel_sz),np.linspace(in_val,fnl_val,kernel_sz),np.linspace(in_val,fnl_val,kernel_sz))
    h=(1/np.sqrt(1.0*2*np.pi*s*s))**(3)*np.exp(-1/(1.0*2*s*s)*(Y**2+X**2+Z**2))
    
    h=np.roll(h,int(grid_nodes/2),axis=0)
    h=np.roll(h,int(grid_nodes/2),axis=1)
    h=np.roll(h,int(grid_nodes/2),axis=2)
    fft_kernel=np.fft.fftn(h)
    fft_img=np.fft.fftn(image)
    smth_img=np.fft.ifftn(np.multiply(fft_kernel,fft_img)).real
    
    plt.figure(figsize=(14,14),dpi=100)
    cmap = plt.get_cmap('jet')#This is where you can change the color scheme   
    plt.title('Smoothed density field')
    if plane==0: dn_fl_plt=plt.imshow(np.power(np.rot90(smth_img[slc,:,:],1),1./scl_plt), cmap=cmap,extent=[0,grid_nodes,0,grid_nodes])
    if plane==1: dn_fl_plt=plt.imshow(np.power(np.rot90(smth_img[:,slc,:],1),1./scl_plt), cmap=cmap,extent=[0,grid_nodes,0,grid_nodes])
    if plane==2: dn_fl_plt=plt.imshow(np.power(np.rot90(smth_img[:,:,slc],1),1./scl_plt), cmap=cmap,extent=[0,grid_nodes,0,grid_nodes])
    plt.colorbar(dn_fl_plt)
    
    plt.savefig('SMOOTHED_DENSITY_FIELD_gd%s_slc%s_smth%sMpc_%splane.png' %(grid_nodes,slc,smooth_scl,plane))
    
    return

def classify_mask(mask,grid_nodes,slc,smooth_scl,plane):
    import numpy as np
    from scipy import ndimage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.colors as mcolors
    
    #The two function below are purely for the color scheme of the imshow plot: Classifier, used to create discrete imshow
    def colorbar_index(ncolors, cmap):
        cmap = cmap_discretize(cmap, ncolors)
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(-0.5, ncolors+0.5)
        colorbar = plt.colorbar(mappable)
        colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
        colorbar.set_ticklabels(range(ncolors))
    
    def cmap_discretize(cmap, N):   
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
        colors_rgba = cmap(colors_i)
        indices = np.linspace(0, 1., N+1)
        cdict = {}
        for ki,key in enumerate(('red','green','blue')):
            cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                           for i in xrange(N+1) ]
        # Return colormap object.
        return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)
    
    #Classifier: This subplot must be first so that the two functions above will help to discretise the color scheme and color bar
    
    fig, ax = plt.subplots(figsize=(14,14),dpi=100) 
    plt.title('LSS Classification')
    plt.xlabel('grid x')
    plt.ylabel('grid y')
    cmap = plt.get_cmap('jet')#This is where you can change the color scheme
    if plane==0: ax.imshow(np.rot90(mask[slc,:,:],1), interpolation='nearest', cmap=cmap,extent=[0,grid_nodes,0,grid_nodes])
    if plane==1: ax.imshow(np.rot90(mask[:,slc,:],1), interpolation='nearest', cmap=cmap,extent=[0,grid_nodes,0,grid_nodes])
    if plane==2: ax.imshow(np.rot90(mask[:,:,slc],1), interpolation='nearest', cmap=cmap,extent=[0,grid_nodes,0,grid_nodes])
    colorbar_index(ncolors=4, cmap=cmap)
    
    plt.savefig('LSS_CLASSIFICATION_gd%d_slc%d_smth%sMpc_%splane.png' %(grid_nodes,slc,smooth_scl,plane))
    
    return

def colorscatter_plt(data,mask,grid_nodes,slc,smooth_scl,plane,sim_sz,plane_thickness,Mass_res,particles_filt,Xc_mult,Xc_minus,Yc_mult,Yc_minus,Zc_mult,Zc_minus):
    import numpy as np
    from matplotlib import pyplot as plt
    import math as mth
    #'data' format reminder:(Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(Mpc/h)*km/s), (Vir. Mass)Mvir(Msun/h) , (Vir. Rad)Rvir(kpc/h) & (color_cd void-0,sheet-1,filament-2,cluster-3)
    partcl_halo_flt=np.where((data[:,9]/(Mass_res))>=particles_filt)#filter for halos with <N particles
    data=data[partcl_halo_flt]#Filter out halos with <N particles
    color_cd=np.zeros(len(data))
    data=np.column_stack((data,color_cd))
    for i in range(len(data)):
       #Create index from halo coordinates
        grid_index_x=mth.trunc(data[i,0]*Xc_mult-Xc_minus)      
        grid_index_y=mth.trunc(data[i,1]*Yc_mult-Yc_minus) 
        grid_index_z=mth.trunc(data[i,2]*Zc_mult-Zc_minus)    
        data[i,11]=mask[grid_index_x,grid_index_y,grid_index_z]
    
    mass=data[:,9]
    rad=data[:,10]    
    #scale mass and radii of halos
    mass=(mass-np.min(mass))/(np.max(mass)-np.min(mass))
    rad=(rad-np.min(rad))/(np.max(rad)-np.min(rad))

    box=np.max(data[:,0])#subset box length
    lo_lim_partcl=1.*slc/(grid_nodes)*box-1.*plane_thickness/2 #For particle distribution
    hi_lim_partcl=lo_lim_partcl+plane_thickness #For particle distributionn
    
    #Filter particles and halos
    partcls=np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    
    for i in range(len(data)):
       
        if (lo_lim_partcl<data[i,plane]<hi_lim_partcl):#incremenets are in 0.00333 #wherver slc is, [x,y,z] make sure corresponds with if statement
            #density field
            result_hals=data[i,:]
            partcls=np.row_stack((partcls,result_hals))
    partcls = np.delete(partcls, (0), axis=0)
    
    #Plotting
    fig, ax = plt.subplots(figsize=(14,14),dpi=100)
    
    i=0#initiate mask for plot loop
    for color in ['red', 'green', 'blue','yellow']:
        lss_plt_filt=np.where(partcls[:,11]==i)
        lss=['voids','sheets','filaments','clusters']  
        scale_factor=1000
        ax.scatter(partcls[lss_plt_filt,0],partcls[lss_plt_filt,2],s=rad*scale_factor,c=color,label=lss[i],alpha=0.9, edgecolors='none')
        i+=1
    
    #ax.view_init(elev=0,azim=-90)#upon generating figure, usually have to rotate manually by 90 deg. clockwise 
    plt.xlabel('x[Mpc/h]') 
    plt.ylabel('y[Mpc/h]')
    plt.title('Color Scatter')
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0,sim_sz])
    ax.set_ylim([0,sim_sz])
    plt.savefig('COLOR_SCATTER_gd%d_slc%d_thck%sMpc_vradius_%splane_%sparticles.png' %(grid_nodes,slc,plane_thickness,plane,particles_filt))
    
    return
    
def alignment_plt(grid_nodes,results):
    from matplotlib import pyplot as plt
    
    plt.figure()
    
    ax2=plt.subplot2grid((1,1), (0,0))
    ax2.axhline(y=0.5, xmin=0, xmax=15, color = 'k',linestyle='--')
    #2000 GRID
    ax2.plot(results[:,0],results[:,2],'g-',label='halo_LSS. 3.5Mpc/h')
    ax2.fill_between(results[:,0], results[:,2]-results[:,3], results[:,2]+results[:,3],facecolor='green',alpha=0.3)
    
    plt.ylabel('Mean cos(theta)')
    plt.xlabel('log Mass[M_solar]')   
    plt.title('Spin-Filament')
    plt.legend(loc='upper right')
    plt.savefig('ALIGNMENT_PLOT_grid_%s.png'%(grid_nodes))
    
    return

def vector_scatter(data,mask,recon_vecs,grid_nodes,sim_sz,particles_filt,slc,Xc_mult,Xc_minus,Yc_mult,Yc_minus,Zc_mult,Zc_minus,plane_thickness):
        
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sklearn.preprocessing as skl
    import math as mth
     
    x_cutout=sim_sz
    z_cutout=sim_sz
    x,y,z=0,1,2
    box=np.max(data[:,0])#subset box length
    partcl_thkns=5#Thickness of the particle slice, Mpc
    lo_lim_partcl=1.*slc/(grid_nodes)*box-1.*partcl_thkns/2 #For particle distribution
    hi_lim_partcl=lo_lim_partcl+partcl_thkns #For particle distributionn
    #Filter halos within slc
    mask_halos=np.zeros(len(data))
    lo_lim_mask=np.where(data[:,y]>lo_lim_partcl)
    hi_lim_mask=np.where(data[:,y]<hi_lim_partcl)
    partcl_500=np.where((data[:,9]/(1.35*10**8))>particles_filt)#filter out halos with <500 particles
    x_mask=np.where(data[:,x]<x_cutout)
    z_mask=np.where(data[:,z]<z_cutout)
    mask_halos[lo_lim_mask]=1
    mask_halos[hi_lim_mask]+=1
    mask_halos[partcl_500]+=1
    mask_halos[x_mask]+=1
    mask_halos[z_mask]+=1
    mask_indx=np.where(mask_halos==5)
    catalog_slc=data[mask_indx]
    
    
    fnl_halos_vecs=[]
    for i in range(len(catalog_slc)):
       #Create index related to the eigenvector bins
        grid_index_x=mth.trunc(catalog_slc[i,0]*Xc_mult-Xc_minus)      
        grid_index_y=mth.trunc(catalog_slc[i,1]*Yc_mult-Yc_minus) 
        grid_index_z=mth.trunc(catalog_slc[i,2]*Zc_mult-Zc_minus) 
        #calculate dot product and bin
        if (mask[grid_index_x,grid_index_y,grid_index_z]==2):#condition includes recon_vecs_unnorm so that I may normalize the vectors which are being processed
            fnl_halos_vecs.append(np.hstack((catalog_slc[i],recon_vecs[grid_index_x,grid_index_y,grid_index_z,0],recon_vecs[grid_index_x,grid_index_y,grid_index_z,1],recon_vecs[grid_index_x,grid_index_y,grid_index_z,2])))
    fnl_halos_vecs=np.asarray(fnl_halos_vecs)
    
    #Plot
    fig, ax = plt.subplots(figsize=(10,30),dpi=350)
    
    #re-normalize projected vectors
    catalog_vec_vel=np.column_stack((fnl_halos_vecs[:,3],fnl_halos_vecs[:,5]))
    catalog_vec_norm_vel=skl.normalize(catalog_vec_vel)
    #Plot velocity vec field 2d
    ax=plt.subplot2grid((3,1), (1,0))
    plt.quiver(fnl_halos_vecs[:,0],fnl_halos_vecs[:,2],catalog_vec_norm_vel[:,0],catalog_vec_norm_vel[:,1],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
    ax.set_xlim([0,x_cutout])
    ax.set_ylim([0,z_cutout])
    plt.xlabel('x[Mpc/h]') 
    plt.ylabel('y[Mpc/h]')
    plt.title('%s velocity vectors'%len(fnl_halos_vecs))
    
    catalog_vec_AM=np.column_stack((fnl_halos_vecs[:,6],fnl_halos_vecs[:,8]))
    catalog_vec_norm_AM=skl.normalize(catalog_vec_AM)
    #Plot AM field 2d
    ax=plt.subplot2grid((3,1), (0,0))
    plt.quiver(fnl_halos_vecs[:,0],fnl_halos_vecs[:,2],catalog_vec_norm_AM[:,0],catalog_vec_norm_AM[:,1],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
    ax.set_xlim([0,x_cutout])
    ax.set_ylim([0,z_cutout])
    plt.xlabel('x[Mpc/h]') 
    plt.ylabel('y[Mpc/h]')
    plt.title('%s Angular Momentum vectors'%len(fnl_halos_vecs))
    
    catalog_vec_eig=np.column_stack((fnl_halos_vecs[:,11],fnl_halos_vecs[:,13]))
    catalog_vec_norm_eig=skl.normalize(catalog_vec_eig)
    #Plot eigvecs field 2d
    ax=plt.subplot2grid((3,1), (2,0))
    plt.quiver(fnl_halos_vecs[:,0],fnl_halos_vecs[:,2],catalog_vec_norm_eig[:,0],catalog_vec_norm_eig[:,1],headwidth=0,minshaft=9,linewidth=0.07,scale=45)
    ax.set_xlim([0,x_cutout])
    ax.set_ylim([0,z_cutout])
    plt.xlabel('x[Mpc/h]') 
    plt.ylabel('y[Mpc/h]')
    plt.title('%s filament axis'%len(fnl_halos_vecs))
    
    plt.savefig('VECTOR_SCATTER_partcls%s_gd%d_slc%d_thck%sMpc_yplane_%s_%s_filament.png' %(particles_filt,grid_nodes,slc,plane_thickness,x_cutout,z_cutout))
