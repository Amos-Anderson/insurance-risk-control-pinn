import torch

def exponential_pdf(z, beta=1.0):
    return beta * torch.exp(-beta * z)

def pareto_pdf(z, alpha=3.0, k=1.0):
    return alpha * k**alpha / (z + k)**(alpha + 1)

def lognormal_pdf(z, mu=0.0, sigma=0.5):
    eps = 1e-8
    z = torch.clamp(z, min=eps)
    two_pi = torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype)
    return (1.0 / (z * sigma * torch.sqrt(two_pi))) * torch.exp(-(torch.log(z) - mu) ** 2 / (2.0 * sigma ** 2))