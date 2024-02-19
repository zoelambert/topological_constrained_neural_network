import os

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from topological_constrained.conf.conf_file import config
from topological_constrained.train.loss import dice_loss, topological_loss
from topological_constrained.train.registration import splitting_algo
from topological_constrained.train.utils import compute_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    dst_train: Dataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
) -> None:
    """
    Function to train the model.

    Args:
        dst_train (Dataset): Training dataset.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): Model to be trained.
    """
    # Optimizer and learning rate scheduler setup
    learning_rate = config["train"]["learning_rate"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=config["train"]["momentum"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", verbose=True)
    checkpoint_dir = config["train"]["checkpoint_dir"]

    # Parameters
    gamma1 = config["train"]["gamma1"]
    gamma2 = config["train"]["gamma2"]
    dt = config["train"]["dt"]
    mu = config["train"]["mu"]
    nu = config["train"]["nu"]

    # Shape of 2D images MxN
    M, N = dst_train[0][0].shape[1], dst_train[0][0].shape[2]

    # Compute constant matrix
    FD, Id, A_N, A_M = compute_params(M, N, DEVICE, gamma1, gamma2, dt)

    for epoch in range(config["train"]["nb_epoch"]):
        train_losses = []
        model.train()

        for i, sample in enumerate(train_loader):
            images, masks = (
                sample[0].to(DEVICE),
                sample[1].long().to(DEVICE),
            )

            optimizer.zero_grad()
            outputs = model(images)

            # Update variable u (with phi=Id+u)
            u = splitting_algo(
                outputs.detach(), masks, FD, Id, A_M, A_N, gamma1, gamma2, dt, mu, nu
            )
            # Update variable theta
            loss = topological_loss(outputs, masks, u, nu)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
            }

        torch.save(
            state,
            os.path.join(
                checkpoint_dir, "sUNet_topological_epoch_{}.pth".format(epoch)
            ),
        )

        model.eval()
        val_losses = []

        for sample in val_loader:
            images, masks = (
                sample[0].to(DEVICE),
                sample[1].long().to(DEVICE),
            )

            outputs = model(images)
            loss = dice_loss(outputs, masks)
            val_losses.append(loss.item())

        scheduler.step(np.mean(val_losses))

        # Print Loss
        print(
            "Epoch: {}. Train Loss: {:.{prec}f}. Val Loss: {:.{prec}f}.".format(
                epoch, np.mean(train_losses), np.mean(val_losses), prec=5
            )
        )
