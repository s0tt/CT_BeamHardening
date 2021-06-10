import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.sampler import SubsetRandomSampler

from utils import parse_json_after_noisy_flags


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        self.num_datasets = len(self.datasets)
        self.len_datasets = [len(d) for d in self.datasets]

    def __getitem__(self, i):
        for j in range(self.num_datasets):
            if i < self.len_datasets[j]: 
                return self.datasets[j][i]
            else: 
                i-= self.len_datasets[j]

    def __len__(self):
        return sum([len(d) for d in self.datasets])


class VolumeDataset(Dataset):
    """Dataset for a single Volume v2, adapted that rotation axis is in the first dimension"""

    def __init__(self, file_path_bh, file_path_gt, num_pixel=256, stride=128, transform=None):
        """
        Args:
            file_path_bh (string): Path to the hdf5 beam hardening volume data.
            file_path_gt (string): Path to the hdf5 ground thruth volume data.
            num_pixel: desired slice size (num_pixel, num_pixel)
            stride: number of pixels which we shift to get the next data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_path_bh = file_path_bh
        self.file_path_gt = file_path_gt
        self.transform = transform
        self.num_pixel = num_pixel
        self.stride = stride

        with h5py.File(self.file_path_bh, 'r') as h5f:
            self.x, self.y, self.z = h5f['Volume'].shape

        self.num_samples_in_y = int((self.y-self.num_pixel)/self.stride) + 1
        self.num_samples_in_z = int((self.z-self.num_pixel)/self.stride) + 1

        self.num_samples_per_slice = self.num_samples_in_y*self.num_samples_in_z

    def __len__(self):
        return self.num_samples_per_slice*(self.x - 4)

    def __getitem__(self, idx):
        x_index = int((idx)/self.num_samples_per_slice) + 2
        overlay =  idx % self.num_samples_per_slice
        z_index = int(overlay/self.num_samples_in_y)
        y_index = overlay % self.num_samples_in_y

        with h5py.File(self.file_path_bh, 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh = volume_bh[x_index-2:x_index+3,
                            y_index*self.stride: y_index*self.stride + self.num_pixel,
                            z_index*self.stride: z_index*self.stride + self.num_pixel]

        with h5py.File(self.file_path_gt, 'r') as h5f:
            volume_gt = h5f['Volume']
            sample_gt = volume_gt[x_index-2:x_index+3,
                            y_index*self.stride: y_index*self.stride + self.num_pixel,
                            z_index*self.stride: z_index*self.stride + self.num_pixel]

        if self.transform:
            sample_gt = self.transform(sample_gt)
            sample_bh = self.transform(sample_bh)

        return [sample_bh, sample_gt]

def get_dataloader(batch_size, number_of_workers, num_pixel, stride, volume_paths, sampler=None,shuffle=True):
    """
        @Args:
            volume_paths: list of tuples, where the first entry is the file path
                        of the .hdf5 file with the beam_hardened volume and the second
                        the .hdf5 file with the ground_truth volume. 
                        [(bh_path, gt_path), ...]
    """

    loader = DataLoader(
                ConcatDataset([VolumeDataset(path[0], path[1], num_pixel, stride) for path in volume_paths]),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=number_of_workers,
                pin_memory=True, # loads them directly in cuda pinned memory 
                drop_last=True,# drop the last incomplete batch
                prefetch_factor=2,# num of (2 * num_workers) samples prefetched
                persistent_workers=False, # keep workers persistent after dataset loaded once
                sampler=sampler # sampler to pass in different indices
                ) 
    return loader


def update_noisy_indexes(num_pixel, dataset_stride, volume_paths, noisy_indexes_path): 
    """
        This function adds the noisy indexes to the noisy_indexes_path file if the noisy indexes
        are not yet known. 

        Note: num_pixel and stride should never be changed after this function was executed once,
        or all noise flags have to set manually to zero 
    """
    dataset_noisy_flags = parse_json_after_noisy_flags(noisy_indexes_path)

    mean_grey_value = 1000
    factor = 1 

    noise_index_file = open(noisy_indexes_path, "r+") 
    noise_index_data = json.load(noise_index_file, encoding="utf-8")



    for index, noisy_flag in enumerate(dataset_noisy_flags): 

        # Check if dataset was already parsed
        if not noisy_flag: 
            dataset = VolumeDataset(volume_paths[index][0], volume_paths[index][1], num_pixel, dataset_stride)
            indexes_to_remove = []

            # iterate over all samples
            for index in range(len(dataset)):
                sample = dataset.__getitem__(index)
                mean_grey_value_sample = (sample.flatten().sum())/sample.size

                # check if sample is just noise
                if mean_grey_value_sample < mean_grey_value*factor: 
                    indexes_to_remove.append(index) 
            
            # add noisy elements to noise_index_data
            for idx, entry in enumerate(noise_index_data["datasets"]): 
                if entry['name'] == volume_paths[index][2]:
                    noise_index_data["datasets"][idx] = {
                                                        "name": entry['name'],
                                                        "noisy_samples_known": True,
                                                        "nr_samples": len(dataset),
                                                        "nr_noisy_samples": len(indexes_to_remove), 
                                                        "noisy_indexes": indexes_to_remove,
                                                        }

    
    json.dump(noise_index_data, noise_index_file)
    noise_index_file.close()


def get_noisy_indexes(noisy_indexes_path: str) -> np.ndarray:

    noisy_indexes = np.array([])
    noise_index_file = open(noisy_indexes_path, "r+") 
    noise_index_data = json.load(noise_index_file, encoding="utf-8")

    offset = 0
    for entry in noise_index_data["datasets"]: 
                # check if sample is just noise
        noisy_indexes = np.concatenate((noisy_indexes, np.array(entry["noisy_indexes"]) + offset))
        offset += entry["nr_samples"]
    
    noise_index_file.close()

    return noisy_indexes




class CtVolumeData(pl.LightningDataModule):
    def __init__(
        self,
        paths,
        path_noise_indexes, 
        batch_size: int = 32,
        num_workers: int = 2,
        dataset_stride: int = 128, 
        num_pixel: int = 256,
        test_split = 0.3,
        val_split = 0.2,
        remove_noisy = None,
        manual_test = None
    ):
        super().__init__()
        self.paths = paths
        self.path_noise_indexes = path_noise_indexes 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_stride = dataset_stride
        self.num_pixel = num_pixel

        loader = get_dataloader(batch_size, num_workers, num_pixel, dataset_stride, 
                                paths, shuffle=False)
        self.dataset_size = len(loader.dataset)
        indices = np.array(range(self.dataset_size))

        remove_idx = np.array([], dtype=int)
        if remove_noisy is not None:
            remove_idx = np.concatenate((remove_noisy, remove_idx))
        
        indices = np.delete(indices, np.unique(remove_idx)) # remove accumulated indices
        split_test = int(np.round(test_split * len(indices)))
        split_val = int(np.round(val_split * len(indices)))
        np.random.shuffle(indices)
        if manual_test is None:
            test_indices = indices[:split_test]
        val_indices = indices[split_test:split_test+split_val]
        train_indices = indices[split_test+split_val:]

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)

    def train_dataloader(self):
        return get_dataloader(self.batch_size, self.num_workers, self.num_pixel, self.dataset_stride, self.paths, 
                                    sampler=self.train_sampler, shuffle=False)

    def val_dataloader(self):
        return get_dataloader(self.batch_size, self.num_workers, self.num_pixel, self.dataset_stride, self.paths, 
                                    sampler=self.val_sampler, shuffle=False)

    def test_dataloader(self, override_batch_size=None):
        if override_batch_size is not None:
            return get_dataloader(override_batch_size, self.num_workers, self.num_pixel, self.dataset_stride, self.paths, 
                                    sampler=self.test_sampler, shuffle=False)
        else:
            return get_dataloader(self.batch_size, self.num_workers, self.num_pixel, self.dataset_stride, self.paths, 
                                    sampler=self.test_sampler, shuffle=False)