import torch 
from tqdm import tqdm
from torch_sparse import spmm
from MCMC.Shrinkage import shrinkage,shrinkage1

def difference(x,axis = 0):
    
    if axis == 0:
        
        diff = x[1:,:]-x[0:-1,:]
        return diff
            
    else:
        
        diff = x[:,1:]-x[:,0:-1]
        return diff
    
        
def inverse_difference(x,axis = 0):
    
    P1,P2 = x.size()
    
    if axis == 0:
        
        inv_diff1 = torch.ones(P1+1,P2,device = x.device)
        inv_diff1[1:-1,:] = x[0:-1,:]-x[1:,:]
        inv_diff1[0,:] = -x[0,:]
        inv_diff1[-1,:] = x[-1,:]
        
        return inv_diff1
    
    else:
        
        inv_diff1 = torch.ones(P1,P2+1,device = x.device)
        inv_diff1[:,1:-1] = x[:,0:-1]-x[:,1:]
        inv_diff1[:,0] = -x[:,0]
        inv_diff1[:,-1] = x[:,-1]
        
        return inv_diff1
        
def sparse_A(A,device):
    
    A_prime = A.to_sparse().to(device)
    A_T = A.T.to_sparse().to(device)

    indice_A = A_prime.indices()
    values_A = A_prime.values()
    
    indice_AT = A_T.indices()
    values_AT = A_T.values()
    
    return indice_A,values_A,indice_AT,values_AT  


def BPS_Gibbs(x_init,Y,A,sigma,hyper,h = 1,M = 300000,burn_in = 300000):
    
    if h == 1: 
        a,b,c,d,e,f = hyper
    else:
        a,b,c,d = hyper
        
    device = Y.device
    N,P = A.shape
    
    indice_A,values_A,indice_AT,values_AT = sparse_A(A,device)
    
    pixel = int(P**0.5)
    sigma2 = sigma**2
    
    x_sample = x_init.to(torch.float32).view(pixel,pixel)
    Y = Y.to(torch.float32)
    A = A.to(torch.float32)
    
    ref = 10
    eta = 100
    
    #Initialization
    
    v = torch.randn(pixel,pixel,device = device)

    D1 = shrinkage1(difference(x_sample,axis = 0),a,b)
    D2 = shrinkage1(difference(x_sample,axis = 1),c,d)
    if h == 1:
        D3 = shrinkage(x_sample, e,f)
        
    x_mean = torch.zeros(pixel,pixel,device = device)
    x_2 = torch.zeros(pixel,pixel,device = device)
    T = 0

    for i in tqdm(range(1,M+burn_in)):
                
        diff1_ax0_v = difference(v,axis = 0)/D1
        diff1_ax1_v = difference(v,axis = 1)/D2
        
        if h == 1:
            
            ink = (spmm(indice_A, values_A, N,P,v.view(-1,1)).square()).sum()/(sigma2*N)+(diff1_ax0_v.square()).sum()/N\
                +(diff1_ax1_v.square()).sum()/N+((v/D3).square()).sum()/N    
                
            gradient = (spmm(indice_AT,values_AT,P,N,(spmm(indice_A, values_A, N,P,x_sample.view(-1,1))-Y))/(N*sigma2)).view(pixel,pixel)\
                +(inverse_difference(difference(x_sample,axis=0)/D1.square(),axis=0)\
                    +inverse_difference(difference(x_sample,axis=1)/D2.square(),axis=1)+x_sample/D3.square())/N
                
        else:
            
            ink = (spmm(indice_A, values_A, N,P,v.view(-1,1)).square()).sum()/(sigma2*N)+(diff1_ax0_v.square()).sum()/N+(diff1_ax1_v.square()).sum()/N    
            
            gradient = (spmm(indice_AT,values_AT,P,N,(spmm(indice_A, values_A, N,P,x_sample.view(-1,1))-Y))/(N*sigma2)).view(pixel,pixel)\
                +(inverse_difference(difference(x_sample,axis=0)/D1.square(),axis=0)\
                    +inverse_difference(difference(x_sample,axis=1)/D2.square(),axis=1))/N
            
        v_gradient = (v*gradient).sum()
        ratio = v_gradient/ink
        
        t3 = float(-ratio+(torch.maximum(ratio,torch.tensor(0,device = device)).square()-2*(torch.rand(1,device = device)).log()/N/ink).sqrt())
        t = torch.tensor([-(torch.rand(1,device=device)).log()/eta,-(torch.rand(1,device = device)).log()/ref,t3])
        
        indicator = torch.argmin(t)
        
        if (i+1) >= burn_in:
            
            x_mean.add_(x_sample, alpha = t[indicator]).add_(v,alpha = 0.5*t[indicator].square())
            x_2.addcmul_(x_sample, x_sample, value = t[indicator]).addcmul_(x_sample,v,value = t[indicator].square())\
                .addcmul(v,v, value = t[indicator].pow(3)/3)
            T += t[indicator]
        
        x_sample.add_(v,alpha = t[indicator])
        
        if indicator == 0:
                        
            D1 = shrinkage1(difference(x_sample,axis = 0),a,b)
            D2 = shrinkage1(difference(x_sample,axis = 1),c,d)
            
            if h == 1:
                D3 = shrinkage(x_sample,e,f)
        
        elif indicator == 1:
            v = torch.randn_like(gradient)
            
        else:
            v.add_(gradient, alpha = - 2*v_gradient/gradient.square().sum())
            
    x_mean = x_mean/T
    x_2 = x_2/T
    x_var = x_2-x_mean.square()
    x_mean[x_mean < 0] = 0
    
    print('Length of the trajectory',T)
    
    return x_mean,x_var.sqrt()
