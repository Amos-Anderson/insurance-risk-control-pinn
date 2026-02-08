import torch

# ------------------------------
# Global model parameters
# ------------------------------
T = 1.0
x_min, x_max = 0.0, 10.0

r = 0.03
mu = 0.08
sigma = 0.20
lambd = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Utility function
# ------------------------------
def utility(x):
    """Exponential utility U(x) = -exp(-x)"""
    return -torch.exp(-x)
