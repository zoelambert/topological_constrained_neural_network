import ctypes
from typing import Tuple

import numpy as np
import torch

from topological_constrained.conf.conf_file import config

# Fonctions en C
lib = ctypes.cdll.LoadLibrary(config["train"]["interpolation_so_file"])
lib.interpol2D.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
]
lib.interpol2D.restype = ctypes.c_void_p

lib.coeff.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="C_CONTIGUOUS"),
]
lib.coeff.restype = ctypes.c_void_p


def interpolation_2D(
    M: int,
    N: int,
    M_R: int,
    N_R: int,
    Id_U0: np.ndarray,
    Id_U1: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fonction d'interpolation 2D.

    Args:
        M (int): Taille de l'image en hauteur.
        N (int): Taille de l'image en largeur.
        M_R (int): Taille de l'image interpolée en hauteur.
        N_R (int): Taille de l'image interpolée en largeur.
        Id_U0 (np.array): Coordonnées de grille de la première dimension.
        Id_U1 (np.array): Coordonnées de grille de la deuxième dimension.
        C (np.array): Coefficients de l'interpolation.

    Returns:
        np.array: Interpolation.
        np.array: Dérivée partielle en x de l'interpolation.
        np.array: Dérivée partielle en y de l'interpolation.
    """
    interpol = np.empty((M, N), dtype=np.float64)
    interpol_dx = np.empty((M, N), dtype=np.float64)
    interpol_dy = np.empty((M, N), dtype=np.float64)

    lib.interpol2D(
        M,
        N,
        M_R,
        N_R,
        Id_U0.ravel(),
        Id_U1.ravel(),
        C.ravel(),
        interpol,
        interpol_dx,
        interpol_dy,
    )

    return interpol, interpol_dx, interpol_dy


def class_interpolation(
    masks: torch.Tensor, U: torch.Tensor, Id: list, nb_class: int = 5
) -> torch.Tensor:
    """
    Fonction pour l'interpolation de classe.

    Args:
        masks (torch.Tensor): Masques.
        U (torch.Tensor): Tenseur.
        Id (list): Coordonnées de grille.
        nb_class (int): Nombre de classes.

    Returns:
        torch.Tensor: Interpolation.
    """
    B, _, M, N = U.shape
    M_R, N_R = M + 20, N + 20
    interpolant = np.zeros((B, 4, M, N))
    for k in range(B):
        for i in range(1, nb_class):
            gt_ext = np.zeros((M_R, N_R), dtype=np.double)
            gt_ext[10:-10, 10:-10] = masks[k] == i
            C = np.zeros((M_R, N_R), dtype=np.double)
            lib.coeff(M_R, N_R, gt_ext.ravel(), C)
            C = np.round(C, 6)

            Id_U0 = Id[0] + U[k, 0].cpu().numpy() + 10
            Id_U1 = Id[1] + U[k, 1].cpu().numpy() + 10
            interpolant_np, _, _ = interpolation_2D(M, N, M_R, N_R, Id_U0, Id_U1, C)

            interpolant[k, i - 1] = interpolant_np.clip(min=0) / np.max(interpolant_np)

    interpolant[np.isnan(interpolant)] = 0

    return torch.from_numpy(np.round(interpolant, 6)).to(device=U.device)


def coefficient_interpolation(masks: torch.Tensor, nb_class: int = 5) -> np.ndarray:
    """
    Fonction pour l'interpolation de coefficient.

    Args:
        masks (torch.Tensor): Masques.
        nb_class (int): Nombre de classes.

    Returns:
        np.array: Coefficient d'interpolation.
    """
    B, M, N = masks.shape
    M_R, N_R = M + 20, N + 20
    coefficient = np.zeros((B, M_R, N_R))
    for i in range(B):
        gt_ext = np.zeros((M_R, N_R), dtype=np.double)
        gt_ext[10:-10, 10:-10] = (1 / (nb_class - 1)) * masks[i]
        C = np.zeros((M_R, N_R), dtype=np.double)
        lib.coeff(M_R, N_R, gt_ext.ravel(), C)
        coefficient[i] = np.round(C, 6)
    return coefficient
