from models.surplus_sde import r, mu, sigma, lambd

def hjb_operator(V_t, V_x, V_xx, V, x, pi, jump_expectation):
    drift = (r * x + pi * (mu - r)) * V_x
    diffusion = 0.5 * (sigma ** 2) * (pi ** 2) * V_xx
    jump = lambd * (jump_expectation - V)
    return V_t + drift + diffusion + jump
