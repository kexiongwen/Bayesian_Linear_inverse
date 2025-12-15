import torch
from MCMC.Fused_L_half.tools import mirror_descent
from MCMC.Fused_L_half.GS_Fused_L_half import Gibbs_sampling_EB
from MCMC.Fused_L_half.BPS_Gibbs import BPS_Gibbs_EB


def EB_Gibbs(x_init, Y, A, sigma, gamma1 = 1, gamma2 = None):
    
    _, P = A.shape
    
    x_sample = x_init.to(torch.float32)
    
    lam1 = 10
    lam2 = 10
    
    if gamma2 is not None:
        
        lam3 = 10
    
    for i in range(30):
        
        if gamma2 is None:
            
            lam1_grad, lam2_grad, x_sample = Gibbs_sampling_EB(x_sample, Y, A, sigma, lam1, lam2, gamma1 = gamma1)
           
        else:
            
            lam1_grad, lam2_grad, lam3_grad, x_sample = Gibbs_sampling_EB(x_sample, Y, A, sigma, lam1, lam2, lam3, gamma1 = gamma1, gamma2 = gamma2)
               
        grad1 = lam1_grad  -  P * 2 ** gamma1 / lam1
        grad2 = lam2_grad  -  P * 2 ** gamma1 / lam2
                
        lam1 = mirror_descent(lam1, grad1)
        lam2 = mirror_descent(lam2, grad2)
        
        print(grad1)
        print(grad2)
        print(lam1)
        print(lam2)
        
        if gamma2 is not None:
            
            grad3 = lam3_grad - P * 2 ** gamma2 / lam3
        
            lam3 = mirror_descent(lam3, grad3)
            
            print(grad3)
            print(lam3)
            
    if gamma2 is None:
        
        x_mean, x_std, _ = Gibbs_sampling_EB(x_init, Y, A, sigma, lam1, lam2, gamma1 = gamma1, EB = False)
        
    else:
        
        x_mean, x_std, _ = Gibbs_sampling_EB(x_init, Y, A, sigma, lam1, lam2, lam3, gamma1 = gamma1, gamma2 = gamma2, EB = False)
    
    return x_mean, x_std
    

def EB_Gibbs_BPS(x_init, Y, A, sigma, gamma1 = 1, gamma2 = None):
    
   
    _, P = A.shape
    
    lam1 = 10
    lam2 = 10
    
    if gamma2 is not None:
        
        lam3 = 10
    
    for i in range(30):
        
        if gamma2 is None:
            
            lam1_grad, lam2_grad = BPS_Gibbs_EB(x_init, Y, A, sigma, lam1 = lam1, lam2 = lam2, gamma1 = gamma1, M = 40000, burn_in = 10000)
        
        else:
            
            lam1_grad, lam2_grad, lam3_grad = BPS_Gibbs_EB(x_init, Y, A, sigma, lam1 = lam1, lam2 = lam2, lam3 = lam3, gamma1 = gamma1, gamma2 = gamma2, M = 40000, burn_in = 10000)
           
        
        grad1 = lam1_grad -  P * 2 ** gamma1 / lam1
        grad2 = lam2_grad -  P * 2 ** gamma1 / lam2
                
        lam1 = mirror_descent(lam1, grad1)
        lam2 = mirror_descent(lam2, grad2)
        
        print(grad1)
        print(grad2)
        print(lam1)
        print(lam2)
        
        if gamma2 is not None:
            
            grad3 = lam3_grad -  P * 2 ** gamma2 / lam3
        
            lam3 = mirror_descent(lam3, grad3)
            
            print(grad3)
            print(lam3)
            
    if gamma2 is None:
        
        x_mean, x_std = BPS_Gibbs_EB(x_init, Y, A, sigma, lam1 = lam1, lam2 = lam2, lam3 = None, gamma1 = gamma1, gamma2 = None, EB = False)
        
    else:
        
        x_mean, x_std = BPS_Gibbs_EB(x_init, Y, A, sigma, lam1 = lam1, lam2 = lam2, lam3 = lam3, gamma1 = gamma1, gamma2 = gamma2, EB = False)
        
    
    return x_mean, x_std