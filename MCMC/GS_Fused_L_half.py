import torch 
from tqdm import tqdm
from MCMC.Shrinkage import shrinkage, shrinkage1
from torch.linalg import solve_triangular

def difference(x, axis = 0):
    
    diff = x.clone()
    
    if axis == 0:
        diff[1:,:] = x[1:,:]-x[0:-1,:]
    else:
        diff[:,1:] = x[:,1:]-x[:,0:-1]
        
    return diff

def Gibbs_sampling(x_init, Y, A, sigma, hyper, gamma1 = 1, gamma2 = None, M = 2500, burn_in = 2500):
    
    x_sample = x_init.to(torch.float32)
    Y = Y.to(torch.float32)
    A = A.to(torch.float32)
    
    if gamma2 is not None:
        a, b, c, d = hyper
    else:
        a, b = hyper
    
    device = A.device
    _,P = A.shape
    pixel = int(P ** 0.5)

    Dv = torch.kron(torch.sparse.spdiags(torch.tensor([[1] * pixel,[-1] * pixel]),torch.tensor([0, -1]), (pixel, pixel)).to_dense(),torch.eye(pixel)).to_sparse().to(device)
    Dh = torch.kron(torch.eye(pixel),torch.sparse.spdiags(torch.tensor([[1] * pixel,[-1] * pixel]),torch.tensor([0, -1]), (pixel, pixel)).to_dense()).to_sparse().to(device)
    
    x_mean = torch.zeros((P, 1), device = device)
    x_2 = torch.zeros((P, 1), device = device)
    
    for i in tqdm(range(1, M + burn_in)):
        
        if gamma1 == 0:
            D1 = shrinkage1(difference(x_sample.view(pixel, pixel), 0), a, b, std = False).view(-1,1)
            D2 = shrinkage1(difference(x_sample.view(pixel, pixel), 1), a, b, std = False).view(-1,1)
        else:
            D1 = shrinkage(difference(x_sample.view(pixel, pixel), 0), a, b, std = False).view(-1,1)
            D2 = shrinkage(difference(x_sample.view(pixel, pixel), 1), a, b, std = False).view(-1,1)
            
        if gamma2 is not None:
            
            if gamma2 == 0:
                D3 = shrinkage1(x_sample, c, d, std = False).view(-1,1)
            else:
                D3 = shrinkage(x_sample, c, d, std = False).view(-1,1)
                
        if gamma2 is not None:
            _,R = torch.linalg.qr(torch.concatenate(((D1 * Dv).to_dense(),(D2 * Dh).to_dense(),torch.diag(D3.ravel()))))
        else:    
            _,R = torch.linalg.qr(torch.concatenate(((D1 * Dv).to_dense(),(D2 * Dh).to_dense())))
            
            
        CAT = solve_triangular(R.T,A.T,upper = False)/sigma
        
        x_sample = solve_triangular(R,torch.linalg.solve(CAT @ CAT.T+torch.eye(P, device = device),CAT @ Y / sigma\
            + CAT @ torch.randn_like(Y) + torch.randn_like(x_mean)), upper = True)

        if (i + 1) > burn_in:
            
            x_mean.add_(x_sample, alpha = 1 / M) 
            x_2.addcmul_(x_sample, x_sample, value = 1 / M)
                    
    x_var = x_2 - x_mean.square()
    x_mean[x_mean < 0] = 0
    
    return x_mean.view(pixel, pixel), x_var.sqrt().view(pixel, pixel)









