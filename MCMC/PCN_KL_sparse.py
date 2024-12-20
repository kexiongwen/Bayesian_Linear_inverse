import torch
import numpy as np
from torch_sparse import spmm
from tqdm import tqdm
from sklearn.gaussian_process.kernels import Matern

def Matern_prior(img_size = 3, length_scale = 0.2, nu = 0.5):
    
    xs = np.linspace(0, 1, img_size + 1)
    delta_x = (xs[1] - xs[0]) / 2
    xs = xs[:-1] + delta_x

    ys = np.linspace(0, 1, img_size + 1)
    ys = ys[:-1] + delta_x

    [X, Y] = np.meshgrid(xs, ys)
    XYpoint = np.c_[X.flatten(), Y.flatten()]

    matern_kernel = Matern(length_scale = length_scale, nu = nu)
    T = matern_kernel(XYpoint)
    return  T

def total_variation(img):
    """Compute total variation statistics."""

    diff1 = img[1:, :] - img[:-1, :]
    diff2 = img[:, 1:] - img[:, :-1]
    tv_value = diff1.abs().sum() + diff2.abs().sum()
    
    return tv_value

def Psi(y, x, indice_A, values_A, N, P, sigma2, pixel, lambda_tv):
    
    res = y - spmm(indice_A, values_A, N, P, x)
    Phi = res.T @ res / (2 * sigma2)
    x_img = x.view(pixel, pixel)
    R = lambda_tv * total_variation(x_img)
    
    return Phi + R

def sparse_A(A):
        
    A_prime = A.to_sparse()

    indice_A = A_prime.indices()
    values_A = A_prime.values()
    
    return indice_A, values_A

def PCN(x_init, Y, A, sigma, hyper, M = 500000, burn_in = 500000):
    
    x = x_init.to(torch.float64)
    Y = Y.to(torch.float64)
    A = A.to(torch.float64)
    
    length_scale, nu, lambda_tv, beta = hyper

    device = A.device
    N,P = A.shape
    pixel = int(P ** 0.5)
    sigma2 = sigma ** 2
    indice_A, values_A = sparse_A(A)
        
    T = Matern_prior(img_size = pixel, length_scale = length_scale, nu = nu)
    eta, E = np.linalg.eigh(T)
    sorted_indices = np.argsort(eta)[::-1]
    eta_decend = eta[sorted_indices]
    total_sum = np.sum(eta_decend)
    cumulative_sum = np.cumsum(eta_decend)
    percentage_threshold = total_sum * 0.95
    K = np.where(cumulative_sum >= percentage_threshold)[0][0]
    print(f"\nthe truncated number of K is {K}: {(K/pixel**2)*100:.2f}%!!!\n")

    eta_truncated = torch.from_numpy(eta[sorted_indices][:K]).to(device)
    E_truncated = torch.from_numpy(E[:, sorted_indices][:, :K]).to(device)
    
    sqrt_Eta_truncated = (eta_truncated).sqrt().view(-1,1)
    
    x_mean = torch.zeros((P, 1), device = device)
    x_square = torch.zeros((P, 1), device = device)
    
    accept = 0
    
    E = torch.from_numpy(E).to(device)
    eta = torch.from_numpy(eta).to(device)

    c = torch.zeros(E.shape[0], device = device, dtype = torch.float64)
    
    for i in range(E.shape[0]):
        c[i] = x.T @ E[:,i:i+1] / eta[i].sqrt()
        
    c0 = c[0:K].view(-1,1) + torch.randn_like(c[0:K].view(-1,1)) * 0.1
    x0 = E_truncated @ (sqrt_Eta_truncated * c0)
    Loss_old = Psi(Y, x0, indice_A, values_A, N, P, sigma2, pixel, lambda_tv)
    
    for i in tqdm(range(M + burn_in)):
    
        c1 = (1 - beta ** 2) ** 0.5 * c0 + beta * torch.randn_like(c0)
        x1 = E_truncated @ (sqrt_Eta_truncated * c1)
        
        Loss_new = Psi(Y, x1, indice_A, values_A, N, P, sigma2, pixel, lambda_tv)
        difference = Loss_old - Loss_new
        
        if (torch.rand(1, device = device)).log() < difference:
            
            c0.copy_(c1)
            x0.copy_(x1)
            Loss_old.copy_(Loss_new) 
            
            if (i+1) > burn_in:
                
                accept += 1 

        if (i+1) > burn_in: 
            
            x_mean.add_(x0, alpha = 1 / M)
            x_square.addcmul_(x0, x0, value = 1 / M)            
            accept_rate = accept / (i + 1 - burn_in) * 100
            
            if i%5000 == 0:
                print(f"\naccepted ratio:{accept_rate}%")
    
    x_var = x_square - x_mean.square()
    x_mean[x_mean < 0] = 0
    
    return x_mean.view(pixel, pixel), x_var.sqrt().view(pixel, pixel)
