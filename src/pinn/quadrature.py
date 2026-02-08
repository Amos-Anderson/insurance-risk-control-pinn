import numpy as np
import torch

def gauss_legendre_quadrature(K, z_max, device):
    xi, wi = np.polynomial.legendre.leggauss(int(K))
    z = 0.5 * (xi + 1.0) * float(z_max)
    w = 0.5 * float(z_max) * wi
    z_t = torch.tensor(z, dtype=torch.float32, device=device)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    return z_t, w_t

def expectation_proportional(model, t, x, rho, pdf_fn, z_k, w_k):
    """
    E[V(t, x - (1-rho)Z)] approx sum w_k V(t, x-(1-rho)z_k) f(z_k)
    t,x,rho: (N,1)
    z_k,w_k: (K,)
    """
    N = x.shape[0]
    K = z_k.shape[0]

    z = z_k.view(K, 1, 1)
    w = w_k.view(K, 1, 1)

    fz = pdf_fn(z)  # (K,1,1)

    x_b = x.view(1, N, 1)
    rho_b = rho.view(1, N, 1)

    x_shift = x_b - (1.0 - rho_b) * z

    t_flat = t.view(1, N, 1).expand(K, N, 1).reshape(K * N, 1)
    x_flat = x_shift.reshape(K * N, 1)

    V_flat, _, _ = model(t_flat, x_flat)
    V_vals = V_flat.view(K, N, 1)

    E_hat = torch.sum(w * fz * V_vals, dim=0)  # (N,1)
    return E_hat

def expectation_xol(model, t, x, M, pdf_fn, z_k, w_k):
    """
    E[V(t, x - min(Z,M))] approx sum w_k V(t, x-min(z_k,M)) f(z_k)
    """
    N = x.shape[0]
    K = z_k.shape[0]

    z = z_k.view(K, 1, 1)
    w = w_k.view(K, 1, 1)

    fz = pdf_fn(z)  # (K,1,1)

    x_b = x.view(1, N, 1)
    M_b = M.view(1, N, 1)

    retained = torch.minimum(z, M_b)  # (K,N,1)
    x_shift = x_b - retained

    t_flat = t.view(1, N, 1).expand(K, N, 1).reshape(K * N, 1)
    x_flat = x_shift.reshape(K * N, 1)

    V_flat, _, _ = model(t_flat, x_flat)
    V_vals = V_flat.view(K, N, 1)

    E_hat = torch.sum(w * fz * V_vals, dim=0)
    return E_hat
