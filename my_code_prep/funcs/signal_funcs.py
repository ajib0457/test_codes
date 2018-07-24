def line(min_bin,max_bin,min_val,max_val):
    import numpy as np
    '''
    The purpose of this function is to calc the gradient and y-intercept of the line connecting two given points
    '''
    #get eqn of line connecting points
    grad=1.*(max_val-min_val)/(max_bin-min_bin)
    y_intcpt=min_val-grad*min_bin
    
    return grad,y_intcpt
    
def subtract_grad(a_x,a_y,b_x,b_y,sig_type):
    import numpy as np
    '''
    The purpose of this function is to take two signals, the first is the fiducial array which 
    means it will be used in calculating the gradient between each of its points.
    '''
    #first identify the extremes    
    no_bins=len(a_y)    
    a_y_grad=[]
    b_x_grad=[]
    b_y_grad=[]
    for i in range(no_bins-1):
        #bin extremes from fiducial data
        min_bin=a_x[i]        
        max_bin=a_x[i+1]        
        min_val=a_y[i]        
        max_val=a_y[i+1]        
        grad,y_intcpt=line(min_bin,max_bin,min_val,max_val)
        #Find points within range                
        points=np.logical_and(b_x>=min_bin,b_x<max_bin)
        if any(points==True):
            mag_points=b_y[points]
            pos_points=b_x[points]
            y_vals=grad*pos_points+y_intcpt
            for i in range(len(y_vals)):
                #save into list
                b_y_grad.append((mag_points[i]))
                b_x_grad.append((pos_points[i]))                                    
                a_y_grad.append((y_vals[i]))   
    #Convert the lists into arrays
    a_y_grad=np.asarray(a_y_grad)
    b_x_grad=np.asarray(b_x_grad)
    b_y_grad=np.asarray(b_y_grad)
    if sig_type=='error':
        a_b_res=abs(a_y_grad-b_y_grad)
        
    if sig_type=='signal':
        a_b_res=abs(a_y_grad-b_y_grad)
    
    return a_b_res,b_x_grad
    
def point_signal_width(a_err_pos,a_err_neg,a_x,x_val):
    import numpy as np
    '''
    This code calculates the width of a signal error at a point x_val
    '''
    a_val_min=np.min(a_x[np.where(a_x>=x_val)])
    if a_val_min==x_val:#Which it will probably half the time
        y_val_a_pos=a_err_pos[np.where(x_val==a_x)]
        y_val_a_neg=a_err_neg[np.where(x_val==a_x)]
        wdth=abs(y_val_a_pos-y_val_a_neg)
           
    else:    
        a_val_max=np.max(a_x[np.where(a_x<x_val)])
    
        #Find the corresponding y_vals    
        a_min_pos=a_err_pos[np.where(a_val_min==a_x)]
        a_max_pos=a_err_pos[np.where(a_val_max==a_x)]
        a_min_neg=a_err_neg[np.where(a_val_min==a_x)]
        a_max_neg=a_err_neg[np.where(a_val_max==a_x)]
        
        #get the gradients & y-intercepts
        grad_a_pos,y_intcpt_a_pos=line(a_val_min,a_val_max,a_min_pos,a_max_pos)
        grad_a_neg,y_intcpt_a_neg=line(a_val_min,a_val_max,a_min_neg,a_max_neg)
        
        #substitute x_val and subtract to get widths
        y_val_a_pos=grad_a_pos*x_val+y_intcpt_a_pos
        y_val_a_neg=grad_a_neg*x_val+y_intcpt_a_neg
        wdth=abs(y_val_a_pos-y_val_a_neg)
    
    return wdth

def calc_resid(a_x,a_y,a_err_pos,a_err_neg,b_x,b_y,b_err_pos,b_err_neg):     
    import numpy as np
    #resid
    lcdm_wdm_res_y,lcdm_wdm_res_x=subtract_grad(a_x,a_y,b_x,b_y,sig_type='signal')
    wdm_lcdm_res_y,wdm_lcdm_res_x=subtract_grad(b_x,b_y,a_x,a_y,sig_type='signal')
    res_tot_y=np.hstack((lcdm_wdm_res_y,wdm_lcdm_res_y))
    res_tot_x=np.hstack((lcdm_wdm_res_x,wdm_lcdm_res_x))
    res=np.column_stack((res_tot_x,res_tot_y))#x,y columns
    res = res[res[:,0].argsort()]
    #Error subtractions
    lcdm_wdm_pn_res_y,lcdm_wdm_pn_res_x=subtract_grad(a_x,a_err_pos,b_x,b_err_neg,sig_type='error')
    wdm_lcdm_np_res_y,wdm_lcdm_np_res_x=subtract_grad(b_x,b_err_neg,a_x,a_err_pos,sig_type='error')
    pn_err_tot_y=np.hstack((lcdm_wdm_pn_res_y,wdm_lcdm_np_res_y))
    pn_err_tot_x=np.hstack((lcdm_wdm_pn_res_x,wdm_lcdm_np_res_x))
    res_lcdm_wdm_pn=np.column_stack((pn_err_tot_x,pn_err_tot_y))#x,y columns
    res_lcdm_wdm_pn=res_lcdm_wdm_pn[res_lcdm_wdm_pn[:,0].argsort()]
    
    lcdm_wdm_np_res_y,lcdm_wdm_np_res_x=subtract_grad(a_x,a_err_neg,b_x,b_err_pos,sig_type='error')
    wdm_lcdm_pn_res_y,wdm_lcdm_pn_res_x=subtract_grad(b_x,b_err_pos,a_x,a_err_neg,sig_type='error')
    np_err_tot_y=np.hstack((lcdm_wdm_np_res_y,wdm_lcdm_pn_res_y))
    np_err_tot_x=np.hstack((lcdm_wdm_np_res_x,wdm_lcdm_pn_res_x))
    res_lcdm_wdm_np=np.column_stack((np_err_tot_x,np_err_tot_y))#x,y columns
    res_lcdm_wdm_np=res_lcdm_wdm_np[res_lcdm_wdm_np[:,0].argsort()]
    
    #pos-neg and neg-pos values for residual
    err_tot=np.vstack((res_lcdm_wdm_np[:,1],res_lcdm_wdm_pn[:,1]))
    
    err_pos=[]
    err_neg=[]
    #This loop finds the maximum and minimum error values for residual
    for i in range(len(res_lcdm_wdm_np)):
        max_val=np.max(err_tot[:,i])
        err_pos.append(max_val)  
        x_val=res_lcdm_wdm_np[i,0]
           
        wdth_a=point_signal_width(a_err_pos,a_err_neg,a_x,x_val)
        wdth_b=point_signal_width(b_err_pos,b_err_neg,b_x,x_val)    
        
        if max_val<=wdth_a+wdth_b:
            min_val=0
            err_neg.append(min_val)
        else:
            err_neg.append(np.min(err_tot[:,i]))
    err_pos=np.asarray(err_pos)
    err_neg=np.asarray(err_neg)
    
    return res,err_pos,err_neg
