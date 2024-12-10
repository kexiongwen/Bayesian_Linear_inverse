import torch 
from torch.distributions.gamma import Gamma

def inv_gauss(mu):
    ink = mu*torch.randn_like(mu).square()
    a = 1+0.5*(ink-((ink+2).square()-4).sqrt())
    return torch.where((1/(1+a))>=torch.rand_like(mu), mu*a, mu/a)

    
def GIG(mu):
    ink=(1e1*torch.randn_like(mu)).square()/(1e2*mu)
    a=1+0.5*(ink-((ink+2).square()-4).sqrt())
    return torch.where((1/(1+a))>=torch.rand_like(mu), mu/a, mu*a)    

def shrinkage(x_sample,a,b, std = True):
    
    N,P = x_sample.size()
    
    #Sample lam
    ink = x_sample.abs().sqrt()
    lam = Gamma(2*N*P+a,ink.sum()+b).sample()
    ink = ink.mul_(lam)
        
    #Sample V
    #v = 2/inv_gauss(1/ink)
    v = 2*GIG(ink)
    
    #Sample tau
    #tau = v/inv_gauss(v/ink.square()).sqrt()
    tau = v*GIG((1e1*ink).square()/(1e2*v)).sqrt()
    
    if std:
        return torch.where(torch.isnan(tau),1e-4,tau/lam.square())
    else:
        return torch.where(torch.isnan(tau), 1e4, torch.where(torch.isinf(tau),1e-4,lam.square()/tau))
        

def shrinkage1(x_sample,a,b, std = True):
    
    N,P = x_sample.size()
    
    #Sample lambda
    ink = x_sample.abs()
    lam = Gamma(N*P+a,ink.sum()+b).sample()
    ink = ink.mul_(lam)      
            
    #Sample tau
    #tau = 1/inv_gauss(1/ink).sqrt()
    tau = GIG(ink).sqrt()
            
    if std:
        return torch.where(torch.isnan(tau),1e-3, tau/lam)
    else:
        return torch.where(torch.isnan(tau), 1e3, torch.where(torch.isinf(tau),1e-3,lam/tau))
        
