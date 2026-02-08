import torch

from models.surplus_sde import T, x_min, x_max, device
from models.hjb_proportional import hjb_operator
from models.hjb_xol import hjb_operator_xol

from pinn.loss import (
    loss_pde,
    loss_terminal,
    loss_boundary,
    loss_reg_proportional,
    loss_reg_xol
)

from pinn.quadrature import (
    expectation_proportional,
    expectation_xol
)

def sample_interior(N, device):
    t = torch.rand(N, 1, device=device) * T
    x = torch.rand(N, 1, device=device) * (x_max - x_min) + x_min
    return t, x

def train_proportional(
    model,
    optimizer,
    pdf_fn,
    z_k,
    w_k,
    num_epochs=3000,
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
):
    model.train()
    loss_history = []

    num_batches = int(N_int // batch_size)
    if num_batches <= 0:
        raise ValueError("N_int must be >= batch_size")

    for epoch in range(1, num_epochs + 1):
        total_loss_epoch = 0.0

        for _ in range(num_batches):
            t, x = sample_interior(batch_size, device)
            t.requires_grad_(True)
            x.requires_grad_(True)

            V, pi, rho = model(t, x)

            ones = torch.ones_like(V)
            V_t = torch.autograd.grad(V, t, grad_outputs=ones, create_graph=True)[0]
            V_x = torch.autograd.grad(V, x, grad_outputs=ones, create_graph=True)[0]
            V_xx = torch.autograd.grad(V_x, x, grad_outputs=ones, create_graph=True)[0]

            E_hat = expectation_proportional(model, t, x, rho, pdf_fn, z_k, w_k)
            residual = hjb_operator(V_t, V_x, V_xx, V, x, pi, E_hat)

            L_pde = loss_pde(residual)
            L_T = loss_terminal(model, device, N_T)
            L_BC = loss_boundary(model, device, N_BC)
            L_reg = loss_reg_proportional(pi, rho, eta_pi=eta_pi, eta_rho=eta_rho)

            total_loss = w_pde * L_pde + w_T * L_T + w_BC * L_BC + L_reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += float(total_loss.item())

        avg_loss = total_loss_epoch / num_batches
        loss_history.append(avg_loss)

        if (epoch % print_every) == 0:
            print("Epoch {:5d} | Avg Loss: {:.4e}".format(epoch, avg_loss))

    return model, loss_history

def train_xol(
    model,
    optimizer,
    pdf_fn,
    z_k,
    w_k,
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
):
    model.train()
    loss_history = []

    num_batches = int(N_int // batch_size)
    if num_batches <= 0:
        raise ValueError("N_int must be >= batch_size")

    for epoch in range(1, num_epochs + 1):
        total_loss_epoch = 0.0

        for _ in range(num_batches):
            t, x = sample_interior(batch_size, device)
            t.requires_grad_(True)
            x.requires_grad_(True)

            V, pi, M = model(t, x)

            ones = torch.ones_like(V)
            V_t = torch.autograd.grad(V, t, grad_outputs=ones, create_graph=True)[0]
            V_x = torch.autograd.grad(V, x, grad_outputs=ones, create_graph=True)[0]
            V_xx = torch.autograd.grad(V_x, x, grad_outputs=ones, create_graph=True)[0]

            E_hat = expectation_xol(model, t, x, M, pdf_fn, z_k, w_k)
            residual = hjb_operator_xol(V_t, V_x, V_xx, V, x, pi, E_hat)

            L_pde = loss_pde(residual)
            L_T = loss_terminal(model, device, N_T)
            L_BC = loss_boundary(model, device, N_BC)
            L_reg = loss_reg_xol(pi, M, M_ref=M_ref, eta_pi=eta_pi, eta_M=eta_M)

            total_loss = w_pde * L_pde + w_T * L_T + w_BC * L_BC + L_reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += float(total_loss.item())

        avg_loss = total_loss_epoch / num_batches
        loss_history.append(avg_loss)

        if (epoch % print_every) == 0:
            print("Epoch {:5d} | Avg Loss: {:.4e}".format(epoch, avg_loss))

    return model, loss_history
