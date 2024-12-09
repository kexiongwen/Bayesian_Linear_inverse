import torch 
from tqdm import tqdm

def difference(x,axis = 0):
    
    if axis == 0:
        
        diff1 = x[1:,:]-x[0:-1,:]
        
        return diff1
            
    else:
        
        diff1 = x[:,1:]-x[:,0:-1]
        
        return diff1

def inverse_difference(x,axis = 0):
    
    P1,P2 = x.size()
    
    if axis == 0:
        
        inv_diff = torch.ones(P1+1,P2,device = x.device)
        inv_diff[1:-1,:] = x[0:-1,:]-x[1:,:]
        inv_diff[0,:] = -x[0,:]
        inv_diff[-1,:] = x[-1,:]
        
        return inv_diff
    
    else:
        
        inv_diff = torch.ones(P1,P2+1,device = x.device)
        inv_diff[:,1:-1] = x[:,0:-1]-x[:,1:]
        inv_diff[:,0] = -x[:,0]
        inv_diff[:,-1] = x[:,-1]
        
        return inv_diff

def proximal_map(x_sample,z1,z2,z3,v1,v2,v3,rho,lambda0,hyper):
    
    eps = 1e-5
    
    lambda1,lambda2 = hyper
    
    # Initialization
    device = x_sample.device
    y = x_sample

    for k in range(0,20):
    
        # Updating y
        
        for i in range(0,20):
            
            gradient = (y-x_sample)/lambda0+rho*inverse_difference((v1+difference(y,axis = 0)-z1),axis = 0)+\
                rho*inverse_difference((v2+difference(y,axis = 1)-z2),axis = 1)+rho*(v3+y-z3)    
                    
            y.add_(gradient,alpha = -eps)
        
        # Updating z
        
        diff_ax0 = difference(y,axis = 0)
        diff_ax1 = difference(y,axis = 1)

        ink1 = diff_ax0+v1
        ink2 = diff_ax1+v2
        ink3 = y+v3
        
        z1 = ink1.sign()*torch.maximum(ink1.abs()-lambda1/rho,torch.tensor(0,device = device))
        z2 = ink2.sign()*torch.maximum(ink2.abs()-lambda1/rho,torch.tensor(0,device = device))
        z3 = ink3.sign()*torch.maximum(ink3.abs()-lambda2/rho,torch.tensor(0,device = device))
                
        # Updating v
        v1.add_(diff_ax0-z1)
        v2.add_(diff_ax1-z2)
        v3.add_(y-z3)
        
    return y,z1,z2,z3,v1,v2,v3

def proximal_map1(x_sample,z1,z2,v1,v2,rho,lambda0,hyper):
    
    eps = 2e-4
    lambda1 = int(hyper)
    
    # Initialization
    device = x_sample.device
    y = x_sample

    for k in range(0,20):
    
        # Updating y
        
        for i in range(0,10):
            
            gradient = (y-x_sample)/lambda0+rho*inverse_difference((v1+difference(y,axis = 0)-z1),axis = 0)+\
                rho*inverse_difference((v2+difference(y,axis = 1)-z2),axis = 1)    
                
            y.add_(gradient,alpha = -eps)
            
        # Updating z
        
        diff_ax0 = difference(y,axis = 0)
        diff_ax1 = difference(y,axis = 1)

        ink1 = diff_ax0+v1
        ink2 = diff_ax1+v2
        
        z1 = ink1.sign()*torch.maximum(ink1.abs()-lambda1/rho,torch.tensor(0,device = device))
        z2 = ink2.sign()*torch.maximum(ink2.abs()-lambda1/rho,torch.tensor(0,device = device))
        
        # Updating v
        v1.add_(diff_ax0-z1)
        v2.add_(diff_ax1-z2)
            
    return y,z1,z2,v1,v2    
    
    
def proximal_langevin(x_init,Y,A,sigma,hyper,h = 1,M = 10000,burn_in = 10000):
    
    
    pixel = int(P**0.5)    
    sigma2 = sigma**2
    device = Y.device
    _,P = A.shape
    
    x_sample = x_init.to(torch.float32).view(pixel,pixel)
    Y = Y.to(torch.float32)
    A = A.to(torch.float32)
    
    rho = 1
    x_mean = torch.zeros(pixel,pixel,device = device)
    x_2 = torch.zeros(pixel,pixel,device = device)
    Lipschitz = torch.linalg.matrix_norm(A.T@A/sigma2,1)
    
    gamma = 1/(6*Lipschitz)
    lambda0 = 1/Lipschitz
    
    z1 = difference(x_sample,axis = 0)
    z2 = difference(x_sample,axis = 1)
    
    if h == 1:
        z3 = x_sample
        v3 = torch.zeros_like(z3)
        
    v1 = torch.zeros_like(z1)
    v2 = torch.zeros_like(z2)
        
    for i in tqdm(range(1,M+burn_in)):
    
        gradient = A.T@(A@x_sample.view(-1,1)-Y).view(pixel,pixel)/sigma2
        
        if h == 1:
            proximal_gradient,z1,z2,z3,v1,v2,v3 = proximal_map(x_sample,z1,z2,z3,v1,v2,v3,rho,lambda0,hyper)
        else:
            proximal_gradient,z1,z2,v1,v2 = proximal_map1(x_sample,z1,z2,v1,v2,rho,lambda0,hyper)
            
        x_sample.mul_(1-gamma/lambda0).add_(gradient, alpha = -gamma).add_(proximal_gradient,alpha = gamma/lambda0)\
            .add_(torch.rand_like(x_sample),alpha = (2*gamma)**0.5)
        
        if (i+1) >= burn_in:
            
            x_mean.add_(x_sample,alpha = 1/M) 
            x_2.addcmul_(x_sample, x_sample, value = 1/M)
            
    x_var = x_2-x_mean.square()
    x_mean[x_mean < 0] = 0

    return x_mean.view(pixel,pixel),x_var.sqrt().view(pixel,pixel)
