import torch
import torch.nn as nn

class HJBPINN(nn.Module):
    def __init__(self, mode="proportional", M_max=10.0, layers=(2, 64, 64, 64, 64, 3)):
        super().__init__()
        if mode not in ("proportional", "xol"):
            raise ValueError("mode must be 'proportional' or 'xol'")
        self.mode = mode
        self.M_max = float(M_max)

        self.activation = nn.Tanh()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def forward(self, t, x):
        z = torch.cat((t, x), dim=1)
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        out = self.layers[-1](z)

        V = out[:, 0:1]
        pi = out[:, 1:2]
        c_raw = out[:, 2:3]

        if self.mode == "proportional":
            rho = torch.sigmoid(c_raw)  # in (0,1)
            return V, pi, rho

        # XoL
        M = self.M_max * torch.sigmoid(c_raw)  # in (0,M_max)
        return V, pi, M
