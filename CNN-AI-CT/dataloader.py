import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import h5py

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

        #return tuple(d[i] for d in self.datasets) # second option return tuple

    def __len__(self):
        return sum([len(d) for d in self.datasets])
        #return min(len(d) for d in self.datasets) # second option minimale size of a volume


class VolumeDataset(Dataset):
    """Dataset for a single Volume"""

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

        self.num_samples_in_x = int((self.x-self.num_pixel)/self.stride) + 1
        self.num_samples_in_z = int((self.z-self.num_pixel)/self.stride) + 1

        self.num_samples_per_slice = self.num_samples_in_z*self.num_samples_in_x  

    def __len__(self):
        return self.num_samples_per_slice*(self.y - 4)

    def __getitem__(self, idx):
        y_index = int((idx)/self.num_samples_per_slice) + 2
        overlay =  idx % self.num_samples_per_slice
        z_index = int(overlay/self.num_samples_in_x)
        x_index = overlay % self.num_samples_in_x

        with h5py.File(self.file_path_bh, 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh = volume_bh[x_index*self.stride: x_index*self.stride + self.num_pixel, 
                            y_index-2:y_index+3, 
                            z_index*self.stride: z_index*self.stride + self.num_pixel]

        with h5py.File(self.file_path_gt, 'r') as h5f:
            volume_gt = h5f['Volume']
            sample_gt = volume_gt[x_index*self.stride: x_index*self.stride + self.num_pixel, 
                            y_index-2:y_index+3, 
                            z_index*self.stride: z_index*self.stride + self.num_pixel]

        if self.transform:
            sample_gt = self.transform(sample_gt)
            sample_bh = self.transform(sample_bh)

        return [sample_bh, sample_gt]


def get_dataloader(batch_size, number_of_gpus, num_pixel, stride, volume_paths):
    """
        @Args:
            volume_paths: list of tuples, where the first entry is the file path
                        of the .hdf5 file with the beam_hardened volume and the second
                        the .hdf5 file with the ground_truth volume. 
                        [(bh_path, gt_path), ...]
    """
    

    train_loader = DataLoader(
                ConcatDataset([VolumeDataset(path[0], path[1], num_pixel, stride) for path in volume_paths]),
                batch_size=batch_size,
                shuffle=True,
                num_workers=number_of_gpus,
                pin_memory=True, # loads them directly in cuda pinned memory 
                drop_last=True) # drop the last incomplete batch

    return train_loader

