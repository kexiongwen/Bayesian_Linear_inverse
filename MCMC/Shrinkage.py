import torch 
from torch.distributions.gamma import Gamma

def inv_gauss(mu):
    ink = mu*torch.randn_like(mu).square()
    a = 1+0.5*(ink-((ink+2).square()-4).sqrt())
    return torch.where((1/(1+a))>=torch.rand_like(mu), mu*a, mu/a)

def shrinkage(x_sample,a,b):
    
    N,P = x_sample.size()
    
    #Sample lam
    ink = x_sample.abs().sqrt()
    lam = Gamma(2*N*P+a,ink.sum()+b).sample()
    ink = ink.mul_(lam)
        
    #Sample V
    v = 2/inv_gauss(1/ink)
    
    #Sample tau
    tau = v/inv_gauss(v/ink.square()).sqrt()
    
    return torch.where(torch.isnan(tau),1000,torch.where(torch.isinf(tau),1000, lam.square()/tau))


def shrinkage1(x_sample,a,b):
    
    N,P = x_sample.size()
    
    #Sample lambda
    ink = x_sample.abs()
    lam = Gamma(N*P+a,ink.sum()+b).sample()
    ink = ink.mul_(lam)      
            
    #Sample tau
    tau = 1/inv_gauss(1/ink).sqrt()
            
    return torch.where(torch.isnan(tau),100,torch.where(torch.isinf(tau),100, lam/tau))
