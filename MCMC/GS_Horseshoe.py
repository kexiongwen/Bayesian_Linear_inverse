import torch 
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from torch.linalg import solve_triangular

def difference(x,axis = 0):
    
    diff = x.clone()
    
    if axis == 0:
        diff[1:,:] = x[1:,:]-x[0:-1,:]
    else:
        diff[:,1:] = x[:,1:]-x[:,0:-1]
        
    return diff


def Gibbs_sampling(x_init,Y,A,sigma,hyper,h = 1,M = 2500,burn_in = 2500):
    
    x_sample = x_init.to(torch.float32)
    Y = Y.to(torch.float32)
    A = A.to(torch.float32)
    
    if h == 1: 
        a,b,c,d,e,f = hyper
    else:
        a,b,c,d = hyper
    
    device = A.device
    _,P = A.size()
    pixel = int(P**0.5)
    x_sample = x_init
    Dv = torch.kron(torch.sparse.spdiags(torch.tensor([[1]*pixel,[-1]*pixel]),torch.tensor([0, -1]), (pixel, pixel)).to_dense(),torch.eye(pixel)).to_sparse().to(device)
    Dh = torch.kron(torch.eye(pixel),torch.sparse.spdiags(torch.tensor([[1]*pixel,[-1]*pixel]),torch.tensor([0, -1]), (pixel, pixel)).to_dense()).to_sparse().to(device)
    
    psi0 = 1
    psi1 = 1
    tau0 = torch.ones(P,device=device)
    tau1 = torch.ones(P,device=device)
    
    v0 = torch.ones(P,device=device)
    v1 = torch.ones(P,device=device)
    
    if h == 1:
        
        psi = 1
        tau = torch.ones(P,device=device)
        v = torch.ones(P,device=device)
    
    x_mean = torch.zeros((P,1),device=device)
    x_2 = torch.zeros((P,1),device=device)

    for i in tqdm(range(1,M+burn_in)):
        
        d0 = difference(x_sample.view(pixel,pixel),0).ravel()
        d1 = difference(x_sample.view(pixel,pixel),1).ravel()
        
        #Sample lambda
        lam0 = (a/psi0+0.5*(d0/tau0).square().sum()/Gamma(torch.tensor([0.5*(a+P)],device=device),1).sample()).sqrt()
        lam1 = (c/psi1+0.5*(d1/tau1).square().sum()/Gamma(torch.tensor([0.5*(c+P)],device=device),1).sample()).sqrt()
        
        #Sample_psi
        psi0 = (1+a/lam0.square())/Gamma(torch.tensor([0.5*(a+1)],device=device),1).sample()
        psi1 = (1+c/lam1.square())/Gamma(torch.tensor([0.5*(c+1)],device=device),1).sample()
        
        #Sample tau2
        tau0 = ((1/v0+0.5*(d0/lam0).square())/Gamma(torch.ones(P,device=device),1).sample()).sqrt()
        tau1 = ((1/v1+0.5*(d1/lam1).square())/Gamma(torch.ones(P,device=device),1).sample()).sqrt()
                
        #Sample V
        v0 = (1/b**2+a/tau0.square())/Gamma(0.5*(a+1)*torch.ones(P,device=device),1).sample()
        v1 = (1/d**2+c/tau1.square())/Gamma(0.5*(c+1)*torch.ones(P,device=device),1).sample()
        
        D1 = 1/(tau0*lam0).reshape(-1,1)
        D2 = 1/(tau1*lam1).reshape(-1,1)
        
        if h == 1:
            
            lam = (e/psi+0.5*(x_sample.ravel()/tau).square().sum()/Gamma(torch.tensor([0.5*(e+P)],device = device),1).sample()).sqrt()
            psi = (1+e/lam.square())/Gamma(torch.tensor([0.5*(e+1)],device = device),1).sample()
            tau = ((1/v+0.5*(x_sample.ravel()/lam).square())/Gamma(torch.ones(P,device = device),1).sample()).sqrt()
            v = (1/f**2+e/tau.square())/Gamma(0.5*(e+1)*torch.ones(P,device = device),1).sample()
            D3 = 1/(tau*lam).ravel()
            _,R = torch.linalg.qr(torch.concatenate(((D1*Dv).to_dense(),(D2*Dh).to_dense(),torch.diag(D3))))
            
        else:
            
            _,R = torch.linalg.qr(torch.concatenate(((D1*Dv).to_dense(),(D2*Dh).to_dense())))
                        
        CAT = solve_triangular(R.T,A.T,upper = False)/sigma
        x_sample = solve_triangular(R,torch.linalg.solve(CAT@CAT.T+torch.eye(P,device=device),CAT@Y/sigma\
            +CAT@torch.randn_like(Y)+torch.randn_like(x_mean)),upper = True)
        
        if (i+1) > burn_in:
            
            x_mean.add_(x_sample,alpha = 1/M) 
            x_2.addcmul_(x_sample, x_sample, value = 1/M)
    
    x_var = x_2-x_mean.square()
    x_mean[x_mean < 0] = 0
                                            
    return x_mean.view(pixel,pixel),x_var.sqrt().view(pixel,pixel)
