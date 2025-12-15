import torch

def difference(x, axis = 0):
    
    if axis == 0:
        
        diff = x[1:,:] - x[0:-1,:]
        
        return diff
            
    else:
        
        diff = x[:,1:] - x[:,0:-1]
        
        return diff
    
def inverse_difference(x, axis = 0):
    
    P1,P2 = x.size()
    
    if axis == 0:
        
        inv_diff = torch.ones(P1 + 1, P2, device = x.device)
        inv_diff[1:-1,:] = x[0:-1,:] - x[1:,:]
        inv_diff[0,:] = -x[0,:]
        inv_diff[-1,:] = x[-1,:]
        
        return inv_diff
    
    else:
        
        inv_diff = torch.ones(P1,P2 + 1,device = x.device)
        inv_diff[:,1:-1] = x[:,0:-1] - x[:,1:]
        inv_diff[:,0] = -x[:,0]
        inv_diff[:,-1] = x[:,-1]
        
        return inv_diff
    
    
def sparse_A(A, device):
    
    A_prime = A.to_sparse().to(device)
    A_T = A.T.to_sparse().to(device)

    indice_A = A_prime.indices()
    values_A = A_prime.values()
    
    indice_AT = A_T.indices()
    values_AT = A_T.values()
    
    return indice_A, values_A, indice_AT, values_AT  

   
def mirror_descent(param, grad, lr = 2e-4):
    
    return param * torch.exp(-lr * grad)

    
def compute_grad_lam(x, v, t, gamma):
    
    alpha = 1 / (2 ** gamma)
    
    b = (x + v * t).abs().pow(alpha)
    
    c = b * t
    
    c[v!=0] = c[v!=0] + (b[v!=0] - x[v!=0].abs().pow(alpha)) *  (x[v!=0] / v[v!=0])
    
    if torch.isnan(c).sum() > 0:
        print('warning: nan grad_lam')
        
    return c.sum() / (alpha + 1)

        