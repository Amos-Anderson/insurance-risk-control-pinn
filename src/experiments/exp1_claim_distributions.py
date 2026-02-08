import torch

from models.surplus_sde import device
from models.claim_distributions import exponential_pdf, pareto_pdf, lognormal_pdf

from pinn.network import HJBPINN
from pinn.quadrature import gauss_legendre_quadrature
from pinn.training import train_proportional

from utils.plotting import plot_loss, evaluate_on_grid, heatmap, plotly_surface

def run_exp1_single(name, pdf_fn, pdf_kwargs, K_quad=32, z_max=10.0, num_epochs=3000):
    # pdf(z) must accept a tensor z and return tensor
    def pdf(z):
        return pdf_fn(z, **pdf_kwargs)

    z_k, w_k = gauss_legendre_quadrature(K_quad, z_max, device)

    model = HJBPINN(mode="proportional", layers=(2, 64, 64, 64, 64, 3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model, history = train_proportional(
        model=model,
        optimizer=optimizer,
        pdf_fn=pdf,
        z_k=z_k,
        w_k=w_k,
        num_epochs=num_epochs,
        N_int=5000,
        batch_size=500,
        N_T=200,
        N_BC=200,
        w_pde=1.0,
        w_T=10.0,
        w_BC=1.0,
        eta_pi=1e-6,
        eta_rho=1e-4,
        print_every=100
    )

    plot_loss(history, title="Experiment {}: Training Loss".format(name))

    t_vals, x_vals, V, pi, rho = evaluate_on_grid(model, t_points=50, x_points=50)
    heatmap(t_vals, x_vals, V, "Experiment {}: V(t,x)".format(name))
    heatmap(t_vals, x_vals, pi, "Experiment {}: pi(t,x)".format(name))
    heatmap(t_vals, x_vals, rho, "Experiment {}: rho(t,x)".format(name))

    # Optional 3D
    # plotly_surface(t_vals, x_vals, V, "Experiment {}: V(t,x)".format(name))
    # plotly_surface(t_vals, x_vals, pi, "Experiment {}: pi(t,x)".format(name))
    # plotly_surface(t_vals, x_vals, rho, "Experiment {}: rho(t,x)".format(name))

    return model, history

def main():
    # 1A: Exponential(beta=1)
    run_exp1_single(
        name="1A Exponential",
        pdf_fn=exponential_pdf,
        pdf_kwargs={"beta": 1.0},
        K_quad=32,
        z_max=10.0,
        num_epochs=3000
    )

    # 1B: Pareto(alpha=3,k=1)
    run_exp1_single(
        name="1B Pareto",
        pdf_fn=pareto_pdf,
        pdf_kwargs={"alpha": 3.0, "k": 1.0},
        K_quad=32,
        z_max=10.0,
        num_epochs=3000
    )

    # 1C: Lognormal(mu=0,sigma=0.5)
    # NOTE: lognormal pdf has z in denominator; quadrature nodes include z>0, ok
    run_exp1_single(
        name="1C Lognormal",
        pdf_fn=lognormal_pdf,
        pdf_kwargs={"mu": 0.0, "sigma": 0.5},
        K_quad=32,
        z_max=10.0,
        num_epochs=3000
    )

if __name__ == "__main__":
    main()
