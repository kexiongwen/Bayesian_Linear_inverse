import torch
from tqdm import tqdm
from torch_sparse import spmm
from MCMC.Fused_L_half.Shrinkage import shrinkage, shrinkage1, shrinkage_EB, shrinkage1_EB
from MCMC.Fused_L_half.tools import difference, inverse_difference, sparse_A,compute_grad_lam


def BPS_Gibbs(x_init, Y, A, sigma, hyper, gamma1 = 1, gamma2 = None, sparse = True,  M = 700000, burn_in = 300000):
        
    if gamma2 is None:
          
        a, b = hyper
        
    else:
        
        a, b, c, d = hyper
    
    device = Y.device
    N, P = A.shape
    
    pixel = int(P ** 0.5)
    sigma2 = sigma ** 2
    
    x_sample = x_init.to(torch.float32).view(pixel,pixel)
    
    Y = Y.to(torch.float32)
    A = A.to(torch.float32)
    
    if sparse:
        
        indice_A, values_A, indice_AT, values_AT = sparse_A(A, device)
    
    ref = 10
    eta = 100
    
    #Initialization
    v = torch.randn(pixel, pixel, device = device)
    
    if sparse:
        
        res = Y - spmm(indice_A, values_A, N, P, x_sample.view(-1,1))
        
    else:
        
        res = Y - A @ x_sample.view(-1,1)
        
    if gamma1 == 0:
            
        D1 = shrinkage1(difference(x_sample, axis = 0), a, b)
        D2 = shrinkage1(difference(x_sample, axis = 1), a, b)
        
    elif gamma1 == 1:
            
        D1 = shrinkage(difference(x_sample, axis = 0), a, b)
        D2 = shrinkage(difference(x_sample, axis = 1), a, b)
    
    if gamma2 is not None:
        
        if gamma2 == 0:
                
            D3 = shrinkage1(x_sample, c, d)
                   
        elif gamma2 == 1:
            
            D3 = shrinkage(x_sample, c, d)
                
        if sparse:
            
            gradient = (spmm(indice_AT, values_AT, P, N, - res) / (N * sigma2)).view(pixel, pixel)\
                    + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                        + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
        
        else:
            
            gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel)\
                + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                    + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
            
    else:
        
        if sparse:
        
            gradient = (spmm(indice_AT, values_AT, P, N, - res) / (N * sigma2)).view(pixel, pixel)\
                    + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                        + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1)) / N
        
        else:
            
            gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel) \
                + (inverse_difference(difference(x_sample, axis = 0) / D1.square(),axis = 0)\
                    + inverse_difference(difference(x_sample, axis = 1) / D2.square(),axis = 1)) / N
                    
    x_mean = torch.zeros(pixel, pixel, device = device)
    x_2 = torch.zeros(pixel, pixel, device = device)
    
    T = 0
    
    for i in tqdm(range(1, M + burn_in)):
                    
        if gamma2 is not None:
            
            if sparse:
            
                ink = (spmm(indice_A, values_A, N, P, v.view(-1,1)).square()).sum() / (sigma2 * N) + ((difference(v,axis = 0) / D1).square()).sum() / N\
                    +((difference(v, axis = 1) / D2).square()).sum() / N + ((v / D3).square()).sum() / N
            
            else:
                
                ink = ((A @ v.view(-1,1)).square()).sum() / (sigma2 * N) + ((difference(v, axis = 0) / D1).square()).sum() / N\
                    + ((difference(v, axis = 1) / D2).square()).sum() / N + ((v / D3).square()).sum() / N 
                        
        else:
            
            if sparse:
            
                ink = (spmm(indice_A, values_A, N, P, v.view(-1,1)).square()).sum() / (sigma2 * N) + ((difference(v, axis = 0) / D1).square()).sum() / N\
                    + ((difference(v, axis = 1) / D2).square()).sum() / N 
                
            else:
                
                ink = ((A @ v.view(-1,1)).square()).sum() / (sigma2 * N) + ((difference(v, axis = 0) / D1).square()).sum() / N\
                    + ((difference(v, axis = 1) / D2).square()).sum() / N 
                
        v_gradient = (v * gradient).sum()
        ratio = v_gradient / ink
        
        t3 = float(- ratio + (torch.maximum(ratio, torch.tensor(0, device = device)).square()-2 * (torch.rand(1, device = device)).log() / N / ink).sqrt())
        t = torch.tensor([-(torch.rand(1, device = device)).log() / eta,-(torch.rand(1,device = device)).log() / ref,t3])
        
        indicator = torch.argmin(t)
        
        if (i+1) >= burn_in:
                
            x_mean.add_(x_sample, alpha = t[indicator]).add_(v,alpha = 0.5 * t[indicator].square())
            x_2.addcmul_(x_sample, x_sample, value = t[indicator]).addcmul_(x_sample,v,value = t[indicator].square())\
                .addcmul(v, v, value = t[indicator].pow(3) / 3)
            
            T += t[indicator]
            
        x_sample.add_(v, alpha = t[indicator])
        
        if sparse:
        
            res = Y - spmm(indice_A, values_A, N, P, x_sample.view(-1,1))
            
        else:
            
            res = Y - A @ x_sample.view(-1,1)

        if indicator == 0:
            
            if gamma1 == 0:
                    
                D1 = shrinkage1(difference(x_sample, axis = 0), a, b)
                D2 = shrinkage1(difference(x_sample, axis = 1), a, b)
                
            elif gamma1 == 1:
                        
                D1 = shrinkage(difference(x_sample, axis = 0), a, b)
                D2 = shrinkage(difference(x_sample, axis = 1), a, b)
                
            if gamma2 is not None:
                
                if gamma2 == 0:
                    
                    D3 = shrinkage1(x_sample, c, d)
                    
                elif gamma2 == 1:
                    
                    D3 = shrinkage(x_sample, c, d)
                        
                if sparse:        
                    
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                    
            else:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1)) / N
                        
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel) \
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(),axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(),axis = 1)) / N
                    
        elif indicator == 1:
            
            v = torch.randn_like(gradient)
            
            if gamma2 is not None:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                        
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                    
            else:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1)) / N
                        
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel) \
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(),axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(),axis = 1)) / N
                    
        else:
            
            if gamma2 is not None:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                    + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                        + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                    
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                    
            else:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1)) / N
                        
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel) \
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(),axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(),axis = 1)) / N
                    
            v_gradient = (v * gradient).sum()
            v.add_(gradient, alpha = - 2 * v_gradient / gradient.square().sum())
            
    print('Length of the trajectory',T)        
    
    x_mean = x_mean / T
    x_2 = x_2 / T
    x_var = (x_2 - x_mean.square()).abs()
    x_mean[x_mean < 0] = 0
        
    return x_mean, x_var.sqrt()


def BPS_Gibbs_EB(x_init, Y, A, sigma, lam1, lam2, lam3 = None, gamma1 = 1, gamma2 = None, sparse = True, EB = True,  M = 70000, burn_in = 30000):
        
    device = Y.device
    N, P = A.shape
    
    pixel = int(P ** 0.5)
    sigma2 = sigma ** 2
    
    x_sample = x_init.to(torch.float32).view(pixel,pixel)
    
    Y = Y.to(torch.float32)
    A = A.to(torch.float32)
    
    if sparse:
        
        indice_A, values_A, indice_AT, values_AT = sparse_A(A, device)
    
    ref = 10
    eta = 100
    
    #Initialization
    v = torch.randn(pixel, pixel, device = device)
    
    if sparse:
        
        res = Y - spmm(indice_A, values_A, N, P, x_sample.view(-1,1))
        
    else:
        
        res = Y - A @ x_sample.view(-1,1)
        
    if gamma1 == 0:
        
        D1 = shrinkage1_EB(difference(x_sample, axis = 0), lam1)
        D2 = shrinkage1_EB(difference(x_sample, axis = 1), lam2)
        
    elif gamma1 == 1:
            
        D1 = shrinkage_EB(difference(x_sample, axis = 0), lam1)
        D2 = shrinkage_EB(difference(x_sample, axis = 1), lam2)
            
    if gamma2 is not None:
        
        if gamma2 == 0:
            
            D3 = shrinkage1_EB(x_sample, lam3)
            
                   
        elif gamma2 == 1:
            
            D3 = shrinkage_EB(x_sample, lam3)
            
        if sparse:
            
            gradient = (spmm(indice_AT, values_AT, P, N, - res) / (N * sigma2)).view(pixel, pixel)\
                    + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                        + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
        
        else:
            
            gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel)\
                + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                    + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
            
    else:
        
        if sparse:
        
            gradient = (spmm(indice_AT, values_AT, P, N, - res) / (N * sigma2)).view(pixel, pixel)\
                    + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                        + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1)) / N
        
        else:
            
            gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel) \
                + (inverse_difference(difference(x_sample, axis = 0) / D1.square(),axis = 0)\
                    + inverse_difference(difference(x_sample, axis = 1) / D2.square(),axis = 1)) / N
    
    if EB:
        
        lam1_grad = 0
        lam2_grad = 0
        
        if gamma2 is not None:
        
            lam3_grad = 0
        
    else:
        
        x_mean = torch.zeros(pixel, pixel, device = device)
        x_2 = torch.zeros(pixel, pixel, device = device)
    
    T = 0
    
    for i in tqdm(range(1, M + burn_in)):
                    
        if gamma2 is not None:
            
            if sparse:
            
                ink = (spmm(indice_A, values_A, N, P, v.view(-1,1)).square()).sum() / (sigma2 * N) + ((difference(v,axis = 0) / D1).square()).sum() / N\
                    +((difference(v, axis = 1) / D2).square()).sum() / N + ((v / D3).square()).sum() / N
            
            else:
                
                ink = ((A @ v.view(-1,1)).square()).sum() / (sigma2 * N) + ((difference(v, axis = 0) / D1).square()).sum() / N\
                    + ((difference(v, axis = 1) / D2).square()).sum() / N + ((v / D3).square()).sum() / N 
                        
        else:
            
            if sparse:
            
                ink = (spmm(indice_A, values_A, N, P, v.view(-1,1)).square()).sum() / (sigma2 * N) + ((difference(v, axis = 0) / D1).square()).sum() / N\
                    + ((difference(v, axis = 1) / D2).square()).sum() / N 
                
            else:
                
                ink = ((A @ v.view(-1,1)).square()).sum() / (sigma2 * N) + ((difference(v, axis = 0) / D1).square()).sum() / N\
                    + ((difference(v, axis = 1) / D2).square()).sum() / N 
                
        v_gradient = (v * gradient).sum()
        ratio = v_gradient / ink
        
        t3 = float(- ratio + (torch.maximum(ratio, torch.tensor(0, device = device)).square()-2 * (torch.rand(1, device = device)).log() / N / ink).sqrt())
        t = torch.tensor([-(torch.rand(1, device = device)).log() / eta,-(torch.rand(1,device = device)).log() / ref,t3])
        
        indicator = torch.argmin(t)
        
        if (i+1) >= burn_in:
            
            if EB:
                
                lam1_grad += compute_grad_lam(difference(x_sample, axis = 0), difference(v, axis = 0), t[indicator], gamma1) 
                lam2_grad += compute_grad_lam(difference(x_sample, axis = 1), difference(v, axis = 1), t[indicator], gamma1)
                 
                if gamma2 is not None:
                    
                    lam3_grad += compute_grad_lam(x_sample, v, t[indicator], gamma2)
                   
            else:
                
                x_mean.add_(x_sample, alpha = t[indicator]).add_(v,alpha = 0.5 * t[indicator].square())
                x_2.addcmul_(x_sample, x_sample, value = t[indicator]).addcmul_(x_sample,v,value = t[indicator].square())\
                    .addcmul(v, v, value = t[indicator].pow(3) / 3)
            
            T += t[indicator]
            
        x_sample.add_(v, alpha = t[indicator])
        
        if sparse:
        
            res = Y - spmm(indice_A, values_A, N, P, x_sample.view(-1,1))
            
        else:
            
            res = Y - A @ x_sample.view(-1,1)

        if indicator == 0:
            
            if gamma1 == 0:
                    
                D1 = shrinkage1_EB(difference(x_sample, axis = 0), lam1)
                D2 = shrinkage1_EB(difference(x_sample, axis = 1), lam2)
                    
                
            elif gamma1 == 1:
                    
                D1 = shrinkage_EB(difference(x_sample, axis = 0), lam1)
                D2 = shrinkage_EB(difference(x_sample, axis = 1), lam2)
                    
            if gamma2 is not None:
                
                if gamma2 == 0:
                        
                    D3 = shrinkage1_EB(x_sample, lam3)
                    
                elif gamma2 == 1:
                        
                    D3 = shrinkage_EB(x_sample, lam3)
                        
                if sparse:        
                    
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                    
            else:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1)) / N
                        
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel) \
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(),axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(),axis = 1)) / N
                    
        elif indicator == 1:
            
            v = torch.randn_like(gradient)
            
            if gamma2 is not None:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                        
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                    
            else:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1)) / N
                        
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel) \
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(),axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(),axis = 1)) / N
                    
        else:
            
            if gamma2 is not None:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                    + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                        + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                    
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1) + x_sample / D3.square()) / N
                    
            else:
                
                if sparse:
                
                    gradient = (spmm(indice_AT, values_AT, P, N, -res) / (N * sigma2)).view(pixel, pixel)\
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(), axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(), axis = 1)) / N
                        
                else:
                    
                    gradient = - (A.T @ res / (N * sigma2)).view(pixel,pixel) \
                        + (inverse_difference(difference(x_sample, axis = 0) / D1.square(),axis = 0)\
                            + inverse_difference(difference(x_sample, axis = 1) / D2.square(),axis = 1)) / N
                    
            v_gradient = (v * gradient).sum()
            v.add_(gradient, alpha = - 2 * v_gradient / gradient.square().sum())
            
    print('Length of the trajectory',T)        
    
    if EB:
            
        lam1_grad = lam1_grad / T 
        lam2_grad = lam2_grad / T 
        
        if gamma2 is not None:
            
            lam3_grad = lam3_grad / T
             
            return lam1_grad, lam2_grad, lam3_grad
        
        else:
            
            return lam1_grad, lam2_grad
            
    else:    
           
        x_mean = x_mean / T
        x_2 = x_2 / T
        x_var = (x_2 - x_mean.square()).abs()
        x_mean[x_mean < 0] = 0
        
        return x_mean, x_var.sqrt()