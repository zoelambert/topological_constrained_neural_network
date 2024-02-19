from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F

from topological_constrained.train.spline_interpolation import *
from topological_constrained.train.utils import dst2D, gradx, grady, newton


def min_u(
    U: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    stheta: torch.Tensor,
    C: torch.Tensor,
    FD: torch.Tensor,
    Id: torch.Tensor,
    A_M: torch.Tensor,
    A_N: torch.Tensor,
    gamma1: float,
    gamma2: float,
    dt: float,
    nu: float,
) -> torch.Tensor:
    """
    Function to minimize U.

    Args:
        U (torch.Tensor): Tensor representing U.
        V (torch.Tensor): Tensor representing V.
        W (torch.Tensor): Tensor representing W.
        stheta (torch.Tensor): Tensor representing softmax of outputs.
        C (torch.Tensor): Tensor representing coefficients for interpolation.
        FD (torch.Tensor): Tensor representing matrix FD.
        Id (torch.Tensor): Tensor representing Id.
        A_M (torch.Tensor): Tensor representing A_M.
        A_N (torch.Tensor): Tensor representing A_N.
        gamma1 (float): Value of gamma1.
        gamma2 (float): Value of gamma2.
        dt (float): Value of dt.
        nu (float): Value of nu.

    Returns:
        torch.Tensor: The updated U tensor.
    """
    B, M, N = stheta.shape
    M_R, N_R = M + 20, N + 20
    div1 = gamma2 * (gradx(V[:, 0]) + grady(V[:, 1])) + gamma1 * (
        gradx(W[:, 0]) + grady(W[:, 1])
    )
    div2 = gamma2 * (gradx(V[:, 2]) + grady(V[:, 3])) + gamma1 * (
        gradx(W[:, 2]) + grady(W[:, 3])
    )

    interpolant = torch.zeros((B, M, N), device=U.device)
    interpolantdx = torch.zeros((B, M, N), device=U.device)
    interpolantdy = torch.zeros((B, M, N), device=U.device)

    for i in range(B):
        Id_U0 = Id[0] + U[i, 0].cpu().numpy() + 10
        Id_U1 = Id[1] + U[i, 1].cpu().numpy() + 10
        interpolant_np, interpolant_dx_np, interpolant_dy_np = interpolation_2D(
            M, N, M_R, N_R, Id_U0, Id_U1, C[i]
        )

        interpolant[i] = torch.from_numpy(np.round(interpolant_np, 6)).to(
            device=U.device
        )
        interpolantdy[i] = torch.from_numpy(np.round(interpolant_dy_np, 6)).to(
            device=U.device
        )
        interpolantdx[i] = torch.from_numpy(np.round(interpolant_dx_np, 6)).to(
            device=U.device
        )

    B1 = U[:, 0] + dt * (
        interpolantdy * 255 * 255 * (-nu * (interpolant - stheta)) - div1
    )
    B2 = U[:, 1] + dt * (
        interpolantdx * 255 * 255 * (-nu * (interpolant - stheta)) - div2
    )

    B1 = B1[:, 1:-1, 1:-1]
    B2 = B2[:, 1:-1, 1:-1]

    U[:, 0, 1:-1, 1:-1] = dst2D(dst2D(B1, A_M, A_N) / FD, A_M, A_N)
    U[:, 1, 1:-1, 1:-1] = dst2D(dst2D(B2, A_M, A_N) / FD, A_M, A_N)

    return U, interpolant


def min_v(V: torch.Tensor, U: torch.Tensor, gamma2: float, mu: float) -> torch.Tensor:
    """
    Function to minimize V.

    Args:
        V (torch.Tensor): Tensor representing V.
        U (torch.Tensor): Tensor representing U.
        gamma2 (float): Value of gamma2.
        mu (float): Value of mu.

    Returns:
        torch.Tensor: The updated V tensor.
    """
    B, _, M, N = U.shape
    b = torch.zeros((B, 4, M, N), device=U.device)
    q = torch.zeros((B, 4, M, N), device=U.device)

    gx_U0, gy_U1 = gradx(U[:, 0]), grady(U[:, 1])
    gx_U1, gy_U0 = gradx(U[:, 1]), grady(U[:, 0])

    b[:, 0] = (gamma2 / sqrt(2)) * (1 + gx_U0 + 1 + gy_U1)
    b[:, 1] = (gamma2 / sqrt(2)) * (1 + gx_U0 - 1 - gy_U1)
    b[:, 2] = (gamma2 / sqrt(2)) * (gy_U0 + gx_U1)
    b[:, 3] = (gamma2 / sqrt(2)) * (gy_U0 - gx_U1)

    coeff_A_hat = (8 * mu + gamma2) / (8 * mu * (b[:, 1] ** 2 + b[:, 2] ** 2))
    coeff_B_hat = 1 / (8 * mu * (b[:, 1] ** 2 + b[:, 2] ** 2))
    w1_hat = torch.pow(
        0.5 * (coeff_B_hat + torch.sqrt(coeff_B_hat**2 + (4 / 27) * coeff_A_hat**3)),
        1 / 3,
    )
    coeff_colinearity_b_hat = w1_hat - coeff_A_hat / (3 * w1_hat)
    # particular case: (b_2^2+b_3^2)=0
    coeff_colinearity_b_hat[torch.isnan(coeff_colinearity_b_hat)] = 1 / (
        gamma2 + 8 * mu
    )

    coeff_A_bar = (-8 * mu + gamma2) / (8 * mu * (b[:, 0] ** 2 + b[:, 3] ** 2))
    coeff_B_bar = 1 / (8 * mu * (b[:, 0] ** 2 + b[:, 3] ** 2))
    w1_bar = torch.pow(
        0.5 * (coeff_B_bar + torch.sqrt(coeff_B_bar**2 + (4 / 27) * coeff_A_bar**3)),
        1 / 3,
    )
    coeff_colinearity_b_bar = w1_bar - coeff_A_bar / (3 * w1_bar)
    # particular case: (b_1^2+b_4^2)=0
    coeff_colinearity_b_bar[torch.isnan(coeff_colinearity_b_bar)] = 1 / (
        gamma2 - 8 * mu
    )

    q[:, 0] = coeff_colinearity_b_bar * b[:, 0]
    q[:, 1] = coeff_colinearity_b_hat * b[:, 1]
    q[:, 2] = coeff_colinearity_b_hat * b[:, 2]
    q[:, 3] = coeff_colinearity_b_bar * b[:, 3]

    V = torch.stack(
        [q[:, 0] + q[:, 1], q[:, 2] + q[:, 3], q[:, 2] - q[:, 3], q[:, 0] - q[:, 1]],
        dim=1,
    ) / sqrt(2)

    # Boundary conditions
    V[:, :, 0, 1:-1] = V[:, :, 1, 1:-1]
    V[:, :, -1, 1:-1] = V[:, :, -2, 1:-1]
    V[:, :, 1:-1, 0] = V[:, :, 1:-1, 1]
    V[:, :, 1:-1, -1] = V[:, :, 1:-1, -2]
    V[:, :, 0, 0] = V[:, :, 1, 1]
    V[:, :, 0, -1] = V[:, :, 1, -2]
    V[:, :, -1, 0] = V[:, :, -2, 1]
    V[:, :, -1, -1] = V[:, :, -2, -2]

    return V


def min_w(
    W: torch.Tensor, U: torch.Tensor, gamma1: float, eps: float = 1e-6
) -> torch.Tensor:
    """
    Function to minimize W.

    Args:
        W (torch.Tensor): Tensor representing W.
        U (torch.Tensor): Tensor representing U.
        gamma1 (float): Value of gamma1.
        eps (float, optional): Value for epsilon. Defaults to 1e-6.

    Returns:
        torch.Tensor: The updated W tensor.
    """
    B, _, M, N = U.shape
    b = torch.zeros((B, 4, M, N), device=U.device)
    y = torch.zeros((B, 4, M, N), device=U.device)

    gx_U0, gy_U1 = gradx(U[:, 0]), grady(U[:, 1])
    gx_U1, gy_U0 = gradx(U[:, 1]), grady(U[:, 0])

    b[:, 0] = (gamma1 / sqrt(2)) * (1 + gx_U0 + 1 + gy_U1)
    b[:, 1] = (gamma1 / sqrt(2)) * (1 + gx_U0 - 1 - gy_U1)
    b[:, 2] = (gamma1 / sqrt(2)) * (gy_U0 + gx_U1)
    b[:, 3] = (gamma1 / sqrt(2)) * (gy_U0 - gx_U1)

    # Compute lagrange multiplier
    lambda_lag = newton(b)

    # Case 1: General case
    condition1 = ((b[:, 0] ** 2 + b[:, 3] ** 2) > eps) * (
        (b[:, 1] ** 2 + b[:, 2] ** 2) > eps
    )
    # Case 2: b_1=b_4=0
    condition2 = ((b[:, 0] ** 2 + b[:, 3] ** 2) < eps) * (
        (b[:, 1] ** 2 + b[:, 2] ** 2) > eps
    )
    # Case 3: b_2=b_3=0 and b_1^2+b_4^2=>8\gamma_1^2
    condition3 = (
        ((b[:, 0] ** 2 + b[:, 3] ** 2) > eps)
        * ((b[:, 1] ** 2 + b[:, 2] ** 2) < eps)
        * ((b[:, 0] ** 2 + b[:, 3] ** 2) >= 8 * gamma1**2)
    )
    # Case 4: b_2=b_3=0 and b_1^2+b_4^2<8\gamma_1^2
    condition4 = (
        ((b[:, 0] ** 2 + b[:, 3] ** 2) > eps)
        * ((b[:, 1] ** 2 + b[:, 2] ** 2) < eps)
        * ((b[:, 0] ** 2 + b[:, 3] ** 2) < 8 * gamma1**2)
    )

    # Compute results depending on conditions
    # Case 1
    y[:, 0][condition1] = b[:, 0][condition1] / (gamma1 + lambda_lag[condition1])
    y[:, 1][condition1] = b[:, 1][condition1] / (gamma1 - lambda_lag[condition1])
    y[:, 2][condition1] = b[:, 2][condition1] / (gamma1 - lambda_lag[condition1])
    y[:, 3][condition1] = b[:, 3][condition1] / (gamma1 + lambda_lag[condition1])
    # Case 2
    y[:, 0][condition2] = (sqrt(2) / 2) * torch.sqrt(
        2 + (b[:, 1][condition2] ** 2 + b[:, 2][condition2] ** 2) / (4 * gamma1**2)
    )
    y[:, 1][condition2] = b[:, 1][condition2] / (2 * gamma1)
    y[:, 2][condition2] = b[:, 2][condition2] / (2 * gamma1)
    y[:, 3][condition2] = (sqrt(2) / 2) * torch.sqrt(
        2 + (b[:, 1][condition2] ** 2 + b[:, 2][condition2] ** 2) / (4 * gamma1**2)
    )
    # Case 3
    y[:, 0][condition3] = b[:, 0][condition3] / (2 * gamma1)
    y[:, 1][condition3] = (sqrt(2) / 2) * torch.sqrt(
        -2 + (b[:, 0][condition3] ** 2 + b[:, 3][condition3] ** 2) / (4 * gamma1**2)
    )
    y[:, 2][condition3] = (sqrt(2) / 2) * torch.sqrt(
        -2 + (b[:, 0][condition3] ** 2 + b[:, 3][condition3] ** 2) / (4 * gamma1**2)
    )
    y[:, 3][condition3] = b[:, 3][condition3] / (2 * gamma1)
    # Case 4
    y[:, 0][condition4] = (
        sqrt(2)
        * b[:, 0][condition4]
        / torch.sqrt(b[:, 0][condition4] ** 2 + b[:, 3][condition4] ** 2)
    )
    y[:, 1][condition4] = 0
    y[:, 2][condition4] = 0
    y[:, 3][condition4] = (
        sqrt(2)
        * b[:, 3][condition4]
        / torch.sqrt(b[:, 0][condition4] ** 2 + b[:, 3][condition4] ** 2)
    )

    W = torch.stack(
        [y[:, 0] + y[:, 1], y[:, 2] + y[:, 3], y[:, 2] - y[:, 3], y[:, 0] - y[:, 1]],
        dim=1,
    ) / sqrt(2)

    # Boundary conditions
    W[:, :, 0, 1:-1] = W[:, :, 1, 1:-1]
    W[:, :, -1, 1:-1] = W[:, :, -2, 1:-1]
    W[:, :, 1:-1, 0] = W[:, :, 1:-1, 1]
    W[:, :, 1:-1, -1] = W[:, :, 1:-1, -2]
    W[:, :, 0, 0] = W[:, :, 1, 1]
    W[:, :, 0, -1] = W[:, :, 1, -2]
    W[:, :, -1, 0] = W[:, :, -2, 1]
    W[:, :, -1, -1] = W[:, :, -2, -2]

    return W


def splitting_algo(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    FD: torch.Tensor,
    Id: torch.Tensor,
    A_M: torch.Tensor,
    A_N: torch.Tensor,
    gamma1: float = 10000,
    gamma2: float = 80000,
    dt: float = 0.01,
    mu: float = 125,
    nu: float = 1,
) -> torch.Tensor:
    """
    Splitting algorithm.

    Args:
        outputs (torch.Tensor): Output tensor.
        masks (torch.Tensor): Tensor of masks.
        FD (torch.Tensor): Tensor representing matrix FD.
        Id (torch.Tensor): Tensor representing Id.
        A_M (torch.Tensor): Tensor representing A_M.
        A_N (torch.Tensor): Tensor representing A_N.
        gamma1 (float, optional): Value of gamma1. Defaults to 10000.
        gamma2 (float, optional): Value of gamma2. Defaults to 80000.
        dt (float, optional): Value of dt. Defaults to 0.01.
        mu (float, optional): Value of mu. Defaults to 125.
        nu (float, optional): Value of nu. Defaults to 1.

    Returns:
        torch.Tensor: The interpolant tensor.
    """
    B, _, M, N = outputs.shape
    stheta = F.softmax(outputs, 1)
    weights = torch.tensor([0.25, 0.5, 0.75, 1.0], device=outputs.device)
    stheta_weighted = torch.sum(stheta[:, 1:] * weights.view(1, -1, 1, 1), dim=1)

    # Initialisation
    U = torch.zeros((B, 2, M, N), device=stheta_weighted.device)
    V = torch.zeros((B, 4, M, N), device=stheta_weighted.device)
    W = torch.zeros((B, 4, M, N), device=stheta_weighted.device)
    C = coefficient_interpolation(masks.cpu())

    for k in range(50):
        V = min_v(V, U, gamma2, mu)
        W = min_w(W, U, gamma1)
        U, interpolant = min_u(
            U, V, W, stheta_weighted, C, FD, Id, A_M, A_N, gamma1, gamma2, dt, nu
        )
    interpolant = class_interpolation(masks.cpu(), U, Id)

    return interpolant
