import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import Subset
import copy
# from torch.utils.data import random_split

from utils import get_mean_grey_value


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        self.num_datasets = len(self.datasets)
        self.len_datasets = [len(d) for d in self.datasets]

    def __getitem__(self, idx):
        i = copy.deepcopy(idx)
        for j in range(self.num_datasets):
            if i < self.len_datasets[j]:
                return idx, self.datasets[j][i]
            else:
                i -= self.len_datasets[j]

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def name_by_index(self, idx):
        i = copy.deepcopy(idx)
        for j in range(self.num_datasets):
            if i < self.len_datasets[j]:
                return self.datasets[j].name
            else:
                i -= self.len_datasets[j]

    def modes_by_indices(self, indices):
        i = copy.deepcopy(indices)
        found_modes = torch.zeros_like(indices)
        material_modes = torch.zeros_like(indices)
        len_cnt = 0

        for j in range(self.num_datasets):
            len_cnt += self.len_datasets[j]
            mask = (i < len_cnt) ^ (found_modes==1)
            if any(mask):
                material_modes[mask] = self.datasets[j].material_mode
                found_modes[mask] = 1
            if sum(found_modes) >= len(indices): break
        
        return material_modes


class VolumeDatasetInfere(Dataset):
    """Inference Dataset for a single Volume, adapted that rotation axis is in the first dimension
        Dataset stride will always be equal to num_pixels"""

    def __init__(self, file_path_bh, num_pixel=256, neighbour_img=[-2, 3], transform=None):
        """
        Args:
            file_path_bh (string): Path to the hdf5 beam hardening volume data.
            num_pixel: desired slice size (num_pixel, num_pixel)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_path_bh = file_path_bh
        self.transform = transform
        self.num_pixel = num_pixel
        self.stride = num_pixel
        self.neighbour_img = neighbour_img

        with h5py.File(self.file_path_bh, 'r') as h5f:
            self.x, self.y, self.z = h5f['Volume'].shape

        self.num_samples_in_y = int((self.y-self.num_pixel)/self.stride) + 1
        self.num_samples_in_z = int((self.z-self.num_pixel)/self.stride) + 1

        self.num_samples_per_slice = self.num_samples_in_y*self.num_samples_in_z

    def __len__(self):
        return self.num_samples_per_slice*(self.x - 4)

    def __getitem__(self, idx):
        x_index = int((idx)/self.num_samples_per_slice) + 2
        overlay = idx % self.num_samples_per_slice
        z_index = int(overlay/self.num_samples_in_y)
        y_index = overlay % self.num_samples_in_y

        with h5py.File(self.file_path_bh, 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh = volume_bh[x_index+self.neighbour_img[0]:x_index+self.neighbour_img[1],
                                  y_index*self.stride: y_index*self.stride + self.num_pixel,
                                  z_index*self.stride: z_index*self.stride + self.num_pixel]

        if self.transform:
            sample_bh = self.transform(sample_bh)

        return sample_bh


class VolumeDatasetTrain(Dataset):
    """Dataset for a single Volume v2, adapted that rotation axis is in the first dimension"""

    def __init__(self, file_path_bh, file_path_gt, num_pixel=256, stride=128, neighbour_img=[-2, 3], transform=None, material_mode=1.0):
        """
        Args:
            file_path_bh (string): Path to the hdf5 beam hardening volume data.
            file_path_gt (string): Path to the hdf5 ground thruth volume data.
            num_pixel: desired slice size (num_pixel, num_pixel)
            stride: number of pixels which we shift to get the next data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name = file_path_bh.split("/")[-3]
        self.file_path_bh = file_path_bh
        self.file_path_gt = file_path_gt
        self.transform = transform
        self.num_pixel = num_pixel
        self.stride = stride
        self.neighbour_img = neighbour_img
        self.material_mode = material_mode

        with h5py.File(self.file_path_bh, 'r') as h5f:
            self.x, self.y, self.z = h5f['Volume'].shape

        self.num_samples_in_y = int((self.y-self.num_pixel)/self.stride) + 1
        self.num_samples_in_z = int((self.z-self.num_pixel)/self.stride) + 1

        self.num_samples_per_slice = self.num_samples_in_y*self.num_samples_in_z

    def __len__(self):
        return self.num_samples_per_slice*(self.x - 4)

    def __getitem__(self, idx):
        x_index = int((idx)/self.num_samples_per_slice) + 2
        overlay = idx % self.num_samples_per_slice
        z_index = int(overlay/self.num_samples_in_y)
        y_index = overlay % self.num_samples_in_y

        with h5py.File(self.file_path_bh, 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh = volume_bh[x_index+self.neighbour_img[0]:x_index+self.neighbour_img[1],
                                  y_index*self.stride: y_index*self.stride + self.num_pixel,
                                  z_index*self.stride: z_index*self.stride + self.num_pixel]

        with h5py.File(self.file_path_gt, 'r') as h5f:
            volume_gt = h5f['Volume']
            sample_gt = volume_gt[x_index+self.neighbour_img[0]:x_index+self.neighbour_img[1],
                                  y_index*self.stride: y_index*self.stride + self.num_pixel,
                                  z_index*self.stride: z_index*self.stride + self.num_pixel]

        if self.transform:
            sample_gt = self.transform(sample_gt)
            sample_bh = self.transform(sample_bh)

        return [sample_bh, sample_gt]


def update_noisy_indexes(num_pixel, dataset_stride, volume_paths, noisy_indexes_path, threshold=1):
    """
        This function adds the noisy indexes to the noisy_indexes_path file if the noisy indexes
        are not yet known. 

        Note: num_pixel and stride should never be changed after this function was executed once,
        or all noise flags have to set manually to zero 
    """

    noise_index_file = open(noisy_indexes_path, "r")
    noise_index_data = json.load(noise_index_file, encoding="utf-8")
    noise_index_file.close()

    for dataset_idx, entry in enumerate(noise_index_data["datasets"]):

        # Check if dataset was already parsed
        if not entry["noisy_samples_known"]:
            dataset = VolumeDatasetTrain(volume_paths[dataset_idx][0],
                                         volume_paths[dataset_idx][1],
                                         num_pixel, dataset_stride)
            indexes_to_remove = []
            # mean_grey_value = get_mean_grey_value(volume_paths[dataset_idx][0]) #(Use maybe later)

            # iterate over all samples
            for idx in range(len(dataset)):
                middle_slice = dataset.__getitem__(idx)[0][2, :, :]
                mean_grey_value_sample = (
                    middle_slice.flatten().sum())/middle_slice.size
                # check if sample is just noise
                if mean_grey_value_sample < threshold:
                    indexes_to_remove.append(idx)

            # add noisy element indexes
            noise_index_data["datasets"][dataset_idx].update({
                "noisy_samples_known": True,
                "nr_samples": len(dataset),
                "nr_noisy_samples": len(indexes_to_remove),
                "noisy_indexes": indexes_to_remove,
            })

    with open(noisy_indexes_path, "w") as noise_index_file:
        json.dump(noise_index_data, noise_index_file, indent=4)


def get_noisy_indexes(noisy_indexes_path: str) -> np.ndarray:

    noisy_indexes = np.array([], dtype=int)
    noise_index_file = open(noisy_indexes_path, "r+")
    noise_index_data = json.load(noise_index_file, encoding="utf-8")

    offset = 0
    for entry in noise_index_data["datasets"]:
        # check if sample is just noise
        noisy_indexes = np.concatenate(
            (noisy_indexes, np.array(entry["noisy_indexes"], dtype=int) + offset))
        offset += entry["nr_samples"]

    noise_index_file.close()

    return noisy_indexes


def custom_random_split(dataset, lengths,
                        generator, remove_idx: np.array):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
        remove_idx (np.array): indices that shall be removed BEFORE split
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    if remove_idx.size:
        indices = np.delete(indices, np.flatnonzero(
            np.isin(indices, remove_idx)))
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class CtVolumeData(pl.LightningDataModule):
    def __init__(
        self,
        paths,
        batch_size: int = 32,
        num_workers: int = 2,
        dataset_stride: int = 128,
        num_pixel: int = 256,
        test_split=0.3,
        val_split=0.2,
        noisy_indexes=None,
        manual_test=None,
        neighbour_img=None,
    ):
        super().__init__()
        self.paths = paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_stride = dataset_stride
        self.num_pixel = num_pixel
        self.test_split = test_split
        self.val_split = val_split
        self.noisy_indexes = noisy_indexes
        self.manual_test = manual_test
        self.neighbour_img = neighbour_img if neighbour_img else [-2, 3]
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...
        self.dataset_seed = 100

        """Split the train and valid dataset"""
        self.dataset_all = ConcatDataset([VolumeDatasetTrain(
            path[0], path[1], self.num_pixel, self.dataset_stride, neighbour_img=self.neighbour_img, material_mode=path[3]) for path in self.paths])
        self.dataset_size = len(self.dataset_all)
        split_test = int(np.round(self.test_split * self.dataset_size))
        split_val = int(np.round(self.val_split * self.dataset_size))
        split_train = int(self.dataset_size - split_test - split_val)

        # remove indices from our samples
        remove_idx = np.array([], dtype=int)
        if noisy_indexes is not None:
            remove_idx = np.concatenate((noisy_indexes, remove_idx))

        # split dataset with seed and remove indices
        self.dataset_train, self.dataset_val, self.dataset_test = custom_random_split(self.dataset_all, [split_train, split_val, split_test],
                                                                                      torch.Generator().manual_seed(self.dataset_seed), remove_idx)

        if remove_idx.size:  # check for success of removal if not empty
            assert np.isin(
                remove_idx, self.dataset_train.indices).all() == False
            assert np.isin(remove_idx, self.dataset_val.indices).all() == False
            assert np.isin(
                remove_idx, self.dataset_test.indices).all() == False

        if self.manual_test is not None:
            random_indices = np.random.randint(
                0, len(self.dataset_test), int(self.manual_test))
            self.dataset_test = Subset(self.dataset_test, random_indices)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # loads them directly in cuda pinned memory
            drop_last=True,  # drop the last incomplete batch
            prefetch_factor=2,  # num of (2 * num_workers) samples prefetched
            persistent_workers=False,  # keep workers persistent after dataset loaded once
            # sampler=self.train_sampler # sampler to pass in different indices
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,  # loads them directly in cuda pinned memory
            drop_last=True,  # drop the last incomplete batch
            prefetch_factor=2,  # num of (2 * num_workers) samples prefetched
            persistent_workers=False,  # keep workers persistent after dataset loaded once
            # sampler=self.val_sampler # sampler to pass in different indices
        )

        return val_loader

    def test_dataloader(self, override_batch_size=None):
        test_loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,  # loads them directly in cuda pinned memory
            drop_last=True,  # drop the last incomplete batch
            prefetch_factor=2,  # num of (2 * num_workers) samples prefetched
            persistent_workers=False,  # keep workers persistent after dataset loaded once
            # sampler=self.test_sampler # sampler to pass in different indices
        )

        return test_loader

    def name_by_index(self, index):
        return self.dataset_all.name_by_index(index)

    def modes_by_indices(self, indices):
        return self.dataset_all.modes_by_indices(indices)
