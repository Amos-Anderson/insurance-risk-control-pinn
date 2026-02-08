import torch
from models.surplus_sde import utility, r, T, x_min, x_max

def loss_pde(residual):
    return torch.mean(residual ** 2)

def loss_terminal(model, device, N_T):
    x_T = torch.rand(N_T, 1, device=device) * (x_max - x_min) + x_min
    t_T = torch.full_like(x_T, T, device=device)
    V_T, _, _ = model(t_T, x_T)
    return torch.mean((V_T - utility(x_T)) ** 2)

def loss_boundary(model, device, N_BC):
    t_L = torch.rand(N_BC, 1, device=device) * T
    x_L = torch.full_like(t_L, x_min)
    V_L, _, _ = model(t_L, x_L)
    B_L = utility(x_L)

    t_R = torch.rand(N_BC, 1, device=device) * T
    x_R = torch.full_like(t_R, x_max)
    x_eff = x_max * torch.exp(r * (T - t_R))
    B_R = utility(x_eff)

    V_R, _, _ = model(t_R, x_R)
    return torch.mean((V_L - B_L) ** 2 + (V_R - B_R) ** 2)

def loss_reg_proportional(pi, rho, eta_pi=1e-6, eta_rho=1e-4):
    return eta_pi * torch.mean(pi ** 2) + eta_rho * torch.mean((rho - 0.5) ** 2)

def loss_reg_xol(pi, M, M_ref=5.0, eta_pi=1e-6, eta_M=1e-4):
    # Mild stabilization; center at an interior reference (change if you want)
    return eta_pi * torch.mean(pi ** 2) + eta_M * torch.mean((M - M_ref) ** 2)
