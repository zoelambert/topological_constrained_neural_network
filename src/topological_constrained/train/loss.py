import torch
import torch.nn.functional as F
from torch import Tensor


def dice_loss(output: Tensor, target: Tensor) -> Tensor:
    """
    Calculate the Dice loss.

    Args:
        output (Tensor): Model predictions.
        target (Tensor): Ground truth labels.

    Returns:
        Tensor: Dice loss.
    """
    target = target.long()
    n, c, h, w = output.size()

    y_onehot = torch.zeros((n, c, h, w), device=output.device)
    y_onehot.scatter_(1, target.view(n, 1, h, w), 1)

    EPSILON = 1e-8
    probs = F.softmax(output, dim=1)[:, 1:]
    y_onehot = y_onehot[:, 1:]

    num = torch.sum(probs * y_onehot, dim=(2, 3))
    den = torch.sum(probs * probs + y_onehot * y_onehot, dim=(2, 3))
    dice = 4 - torch.sum((2 * num + EPSILON) / (den + EPSILON)) / n

    return dice


def topological_loss(output: Tensor, target: Tensor, u: Tensor, nu: float) -> Tensor:
    """
    Calculate the topological loss.

    Args:
        output (Tensor): Model predictions.
        target (Tensor): Ground truth labels.
        u (Tensor): Tensor representing u.
        nu (float): Value of nu.

    Returns:
        Tensor: Topological loss.
    """
    target = target.long()
    n, c, h, w = output.size()

    y_onehot = torch.zeros((n, c, h, w), device=output.device)
    y_onehot.scatter_(1, target.view(n, 1, h, w), 1)

    EPSILON = 1e-8
    probs = F.softmax(output, dim=1)[:, 1:]
    y_onehot = y_onehot[:, 1:]

    num = torch.sum(probs * y_onehot, dim=(2, 3))
    den = torch.sum(probs * probs + y_onehot * y_onehot, dim=(2, 3))
    dice = 4 - torch.sum((2 * num + EPSILON) / (den + EPSILON)) / n

    topo = torch.sum((nu / 2) * (probs - u) * (probs - u)) / (n * h * w)

    loss = torch.mean(topo + dice)

    return loss
