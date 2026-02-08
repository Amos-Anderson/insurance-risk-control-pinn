import numpy as np
import torch
import matplotlib.pyplot as plt

from models.surplus_sde import T, x_min, x_max, device

def plot_loss(history, title="Training Loss"):
    plt.figure(figsize=(7, 5))
    plt.plot(history, label="loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def evaluate_on_grid(model, t_points=50, x_points=50):
    t_vals = np.linspace(0.0, T, int(t_points))
    x_vals = np.linspace(x_min, x_max, int(x_points))

    T_grid, X_grid = np.meshgrid(t_vals, x_vals, indexing="ij")
    t_tensor = torch.tensor(T_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    x_tensor = torch.tensor(X_grid.reshape(-1, 1), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        out1, out2, out3 = model(t_tensor, x_tensor)

    out1 = out1.cpu().numpy().reshape(int(t_points), int(x_points))
    out2 = out2.cpu().numpy().reshape(int(t_points), int(x_points))
    out3 = out3.cpu().numpy().reshape(int(t_points), int(x_points))

    return t_vals, x_vals, out1, out2, out3

def heatmap(t_vals, x_vals, Z, title):
    T_grid, X_grid = np.meshgrid(t_vals, x_vals, indexing="ij")
    plt.figure(figsize=(6, 5))
    plt.contourf(T_grid, X_grid, Z, 40)
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(title)
    plt.show()

def plotly_surface(t_vals, x_vals, Z, title):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError("plotly is required for 3D surfaces: pip install plotly") from e

    T_grid, X_grid = np.meshgrid(t_vals, x_vals, indexing="ij")

    fig = go.Figure(
        data=[
            go.Surface(
                x=T_grid,
                y=X_grid,
                z=Z,
                showscale=False
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="t"),
            yaxis=dict(title="x"),
            zaxis=dict(title=title)
        ),
        autosize=True,
        width=800,
        height=600
    )
    fig.show()
