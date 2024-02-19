from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader

from topological_constrained.conf.conf_file import config


class CTDataset(data.Dataset):
    """
    Dataset class for CT data.
    """

    def __init__(self, files: List[str]) -> None:
        """
        Initialize the CT dataset.

        Args:
            files (List[str]): List of paths to HDF5 files.
        """
        super(CTDataset, self).__init__()
        self.datasets: List[np.ndarray] = []
        self.datasets_gt: List[np.ndarray] = []
        self.total_count: int = 0
        self.limits: List[int] = []
        self.files: List[str] = files

        for file_path in files:
            with h5py.File(file_path, "r") as h5_file:
                dataset = h5_file["data"]
                dataset_gt = h5_file["label"]
                self.datasets.append(dataset)
                self.datasets_gt.append(dataset_gt)
                self.limits.append(self.total_count)
                self.total_count += len(dataset)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Get item by index.

        Args:
            index (int): Index of the item.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: A tuple containing img, label, and index.
        """
        dataset_index = np.searchsorted(self.limits, index, side="right") - 1
        assert dataset_index >= 0, "Negative chunk index"

        in_dataset_index = index - self.limits[dataset_index]
        img, label = self.transform(
            self.datasets[dataset_index][in_dataset_index],
            self.datasets_gt[dataset_index][in_dataset_index],
        )
        return img, label, index

    def __len__(self) -> int:
        """
        Get the total length of the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return self.total_count

    @staticmethod
    def transform(
        img: np.ndarray, label: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform the data.

        Args:
            img (np.ndarray): Input image data.
            label (np.ndarray): Label data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing transformed image and label.
        """
        label = label.astype(np.int)
        label = torch.from_numpy(label).long()
        img_ = torch.from_numpy(img).float()
        return img_, label


def get_data():
    """
    Get training and validation data loaders.

    Returns:
        tuple: A tuple containing the training dataset, training data loader, and validation data loader.
               (train_dataset, train_loader, val_loader)

    Raises:
        FileNotFoundError: If HDF5 files are not found in the specified directories.
    """
    data_dir = config["dataset"]["data_dir"]
    train_path = config["dataset"]["train_dir"]
    val_path = config["dataset"]["validation_dir"]
    batch_size = config["dataset"]["batch_size"]

    hdf5_list_train = list(Path(data_dir, train_path).glob("*.h5"))
    if not hdf5_list_train:
        raise FileNotFoundError(f"No HDF5 files found in {train_path}")

    hdf5_list_val = list(Path(data_dir, val_path).glob("*.h5"))
    if not hdf5_list_val:
        raise FileNotFoundError(f"No HDF5 files found in {val_path}")

    train_dataset = CTDataset(hdf5_list_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    val_dataset = CTDataset(hdf5_list_val)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    return train_dataset, train_loader, val_loader
