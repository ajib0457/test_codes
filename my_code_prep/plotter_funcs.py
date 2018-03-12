
def scatter_plot(Xc,Yc,Zc,slc,plane,grid_nodes,plane_thickness):   
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
    fig, ax = plt.subplots(figsize=(20,20),dpi=100)
    ax.scatter(partcls[:,0],partcls[:,2],c='r')
    ax.set_xlim([0,grid_nodes])
    ax.set_ylim([0,grid_nodes]) 
    plt.xlabel('x[Mpc/h]') 
    plt.ylabel('y[Mpc/h]')
  
    ax.grid(True)
    plt.savefig('bolchoi_halosall_gd%d_slc%d_thck%sMpc_%splane.png' %(grid_nodes,slc,plane_thickness,plane))

def smoothed_density_field(grid_nodes,slc,smooth_scl,in_val,fnl_val,s,image,plane,scl_plt):
    import numpy as np
    from scipy import ndimage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.colors as mcolors
    
    kernel_sz=grid_nodes
    X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,kernel_sz),np.linspace(in_val,fnl_val,kernel_sz),np.linspace(in_val,fnl_val,kernel_sz))
    h=(1/np.sqrt(1.0*2*np.pi*s*s))**(3)*np.exp(-1/(1.0*2*s*s)*(Y**2+X**2+Z**2))

    h=np.roll(h,int(grid_nodes/2),axis=0)
    h=np.roll(h,int(grid_nodes/2),axis=1)
    h=np.roll(h,int(grid_nodes/2),axis=2)
    fft_kernel=np.fft.fftn(h)
    fft_img=np.fft.fftn(image)
    smth_img=np.fft.ifftn(np.multiply(fft_kernel,fft_img)).real

    plt.figure(figsize=(20,20),dpi=100)
    
    #Density field
    ax5=plt.subplot2grid((1,1), (1,0))    
    plt.title('classified image')
    cmapp = plt.get_cmap('jet')
    if plane==0: dn_fl_plt=ax5.imshow(np.power(np.rot90(smth_img[slc,:,:],1),1./scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])
    if plane==1: dn_fl_plt=ax5.imshow(np.power(np.rot90(smth_img[:,slc,:],1),1./scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])
    if plane==2: dn_fl_plt=ax5.imshow(np.power(np.rot90(smth_img[:,:,slc],1),1./scl_plt),cmap=cmapp,extent=[0,grid_nodes,0,grid_nodes])
    plt.colorbar(dn_fl_plt,cmap=cmapp)
    
    plt.savefig('bolchoi_smthden_gd%d_slc%d_smth%sMpc_%splane.png' %(grid_nodes,slc,smooth_scl,plane))
    
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
    ax=plt.subplot2grid((1,1), (0,0))  
    plt.title('Classifier')
    #plt.xlabel('z')
    #plt.ylabel('x')
    cmap = plt.get_cmap('jet')#This is where you can change the color scheme
    if plane==0: ax.imshow(np.rot90(mask[slc,:,:],1), interpolation='nearest', cmap=cmap,extent=[0,grid_nodes,0,grid_nodes])
    if plane==1: ax.imshow(np.rot90(mask[:,slc,:],1), interpolation='nearest', cmap=cmap,extent=[0,grid_nodes,0,grid_nodes])
    if plane==2: ax.imshow(np.rot90(mask[:,:,slc],1), interpolation='nearest', cmap=cmap,extent=[0,grid_nodes,0,grid_nodes])
    colorbar_index(ncolors=4, cmap=cmap)
    
    plt.savefig('bolchoi_recon_img_gd%d_slc%d_smth%sMpc_%splane.png' %(grid_nodes,slc,smooth_scl,plane))
