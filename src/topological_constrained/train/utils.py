from math import pi, sqrt

import numpy as np
import torch


def compute_params(
    M: int,
    N: int,
    device: torch.device,
    gamma1: float = 10000,
    gamma2: float = 80000,
    dt: float = 0.01,
):
    """
    Compute parameters.

    Args:
        M (int): Dimension M.
        N (int): Dimension N.
        device (torch.device): Device to use.
        gamma1 (float, optional): Gamma 1 value. Defaults to 10000.
        gamma2 (float, optional): Gamma 2 value. Defaults to 80000.
        dt (float, optional): Time step. Defaults to 0.01.

    Returns:
        Tuple: Tuple containing computed parameters.
    """
    # Compute matrix FD
    cos_M = torch.cos(pi * torch.arange(1, M - 1) / (M - 1))
    cos_N = torch.cos(pi * torch.arange(1, N - 1) / (N - 1))
    outer_cos = torch.ger(cos_M, torch.ones(N - 2)) + torch.ger(
        torch.ones(M - 2), cos_N
    )
    FD = 1 + 4 * dt * (gamma1 + gamma2) - 2 * dt * (gamma1 + gamma2) * outer_cos
    FD.to(device)

    # Compute field Id
    Id1 = (np.arange(1, M + 1) * np.ones((N, 1))).T
    Id2 = np.ones((M, 1)) * np.arange(1, N + 1)
    Id = np.zeros((2, M, N))
    Id[0], Id[1] = Id1, Id2

    # Compute A_N and A_M
    A_N = torch.sin(
        pi / (N - 1) * (torch.arange(N - 2) + 1).view(-1, 1) * (torch.arange(N - 2) + 1)
    ).to(device)
    A_M = torch.sin(
        pi / (M - 1) * (torch.arange(M - 2) + 1).view(-1, 1) * (torch.arange(M - 2) + 1)
    ).to(device)

    return FD, Id, A_N, A_M


def gradx(U: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient in the x-direction.

    Args:
        U (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Gradient tensor in the x-direction.
    """
    dx = torch.zeros_like(U)
    dx[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / 2
    dx[:, 0] = U[:, 1] - U[:, 0]  # boundary conditions
    dx[:, -1] = U[:, -1] - U[:, -2]
    return dx


def grady(U: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient in the y-direction.

    Args:
        U (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Gradient tensor in the y-direction.
    """
    dy = torch.zeros_like(U)
    dy[:, :, 1:-1] = (U[:, :, 2:] - U[:, :, :-2]) / 2
    dy[:, :, 0] = U[:, :, 1] - U[:, :, 0]  # boundary conditions
    dy[:, :, -1] = U[:, :, -1] - U[:, :, -2]
    return dy


def dst2D(R: torch.Tensor, Am: torch.Tensor, An: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D discrete sine transform.

    Args:
        R (torch.Tensor): Input tensor.
        Am (torch.Tensor): Tensor.
        An (torch.Tensor): Tensor.

    Returns:
        torch.Tensor: Result of the 2D discrete sine transform.
    """
    B, M, N = R.shape
    b = torch.mul(An, R.unsqueeze(3)).sum(2) * (2 / sqrt(2 * (N + 1)))
    result = torch.mul(Am, (b.permute(0, 2, 1)).unsqueeze(3)).sum(2) * (
        2 / sqrt(2 * (M + 1))
    )
    return result.permute(0, 2, 1)


def newton(
    b: torch.Tensor, tol: float = 0.0001, max_iter: int = 20, gamma1: float = 10000
) -> torch.Tensor:
    """
    Apply the Newton method.

    Args:
        b (torch.Tensor): Input tensor.
        tol (float, optional): Tolerance. Defaults to 0.0001.
        max_iter (int, optional): Maximum number of iterations. Defaults to 20.
        gamma1 (float, optional): Gamma 1 value. Defaults to 10000.

    Returns:
        torch.Tensor: Result of the Newton method.
    """
    B, _, M, N = b.shape
    fx = torch.ones((B, M, N), device=b.device)
    grid = torch.arange(-gamma1, gamma1, 100, device=b.device, dtype=torch.float32)
    grid[0] = -gamma1 + 1

    g1 = (b[:, 0] ** 2 + b[:, 3] ** 2).unsqueeze(3) / ((gamma1 + grid) ** 2).reshape(
        1, 1, 1, grid.shape[0]
    )
    g2 = (b[:, 1] ** 2 + b[:, 2] ** 2).unsqueeze(3) / ((gamma1 - grid) ** 2).reshape(
        1, 1, 1, grid.shape[0]
    )

    x_prec = grid[torch.argmin(torch.abs(2 - g1 + g2), 3)]
    x_init = x_prec.clone()

    nb_iter = 0
    while torch.any(torch.abs(fx) > tol) and nb_iter < max_iter:
        nb_iter += 1
        fx = (
            2
            - (b[:, 0] ** 2 + b[:, 3] ** 2) / ((x_prec + gamma1) ** 2)
            + (b[:, 1] ** 2 + b[:, 2] ** 2) / ((gamma1 - x_prec) ** 2)
        )

        jacobian = 2 * (b[:, 0] ** 2 + b[:, 3] ** 2) / ((x_prec + gamma1) ** 3) + 2 * (
            b[:, 1] ** 2 + b[:, 2] ** 2
        ) / ((gamma1 - x_prec) ** 3)

        x_prec = x_prec - fx / jacobian

    x_prec = torch.where(torch.abs(x_prec) > gamma1, x_init.float(), x_prec)

    return x_prec
