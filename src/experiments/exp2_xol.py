import torch

from models.surplus_sde import device
from models.claim_distributions import exponential_pdf

from pinn.network import HJBPINN
from pinn.quadrature import gauss_legendre_quadrature
from pinn.training import train_xol

from utils.plotting import plot_loss, evaluate_on_grid, heatmap, plotly_surface

def main():
    def pdf(z):
        return exponential_pdf(z, beta=1.0)

    K_quad = 32
    z_max = 10.0
    z_k, w_k = gauss_legendre_quadrature(K_quad, z_max, device)

    model = HJBPINN(mode="xol", M_max=10.0, layers=(2, 64, 64, 64, 64, 3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model, history = train_xol(
        model=model,
        optimizer=optimizer,
        pdf_fn=pdf,
        z_k=z_k,
        w_k=w_k,
        num_epochs=3000,
        N_int=5000,
        batch_size=500,
        N_T=200,
        N_BC=200,
        w_pde=1.0,
        w_T=10.0,
        w_BC=1.0,
        eta_pi=1e-6,
        eta_M=1e-4,
        M_ref=5.0,
        print_every=100
    )

    plot_loss(history, title="Experiment 2B XoL: Training Loss")

    t_vals, x_vals, V, pi, M = evaluate_on_grid(model, t_points=50, x_points=50)
    heatmap(t_vals, x_vals, V, "Experiment 2B XoL: V(t,x)")
    heatmap(t_vals, x_vals, pi, "Experiment 2B XoL: pi(t,x)")
    heatmap(t_vals, x_vals, M, "Experiment 2B XoL: M(t,x)")

    # Optional 3D
    # plotly_surface(t_vals, x_vals, V, "Experiment 2B XoL: V(t,x)")
    # plotly_surface(t_vals, x_vals, pi, "Experiment 2B XoL: pi(t,x)")
    # plotly_surface(t_vals, x_vals, M, "Experiment 2B XoL: M(t,x)")

if __name__ == "__main__":
    main()
