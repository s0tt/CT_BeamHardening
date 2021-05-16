import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import h5py


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class VolumeDataset(Dataset):
    """Dataset for a single Volume"""

    def __init__(self, file_path, stride=128, transform=None):
        """
        Args:
            file_path (string): Path to the hdf5 volume data.
            stride: number of pixels in each shift 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_path = file_path
        self.transform = transform
        self.stride = stride

        with h5py.File(self.file_path, 'r') as h5f:
            self.m, self.k, self.n = h5f['Volume'].shape

        self.num_samples_per_slice = (int(self.n/self.stride) - 2)*(int(self.m/self.stride) - 2) 

    def __len__(self):
        return self.num_samples_per_slice*(self.k - 4)

    def __getitem__(self, idx):

        """
            At the moment we do just support single indexes, no lists
        if torch.is_tensor(idx):
            idx = idx.tolist()
        """

        y_index = int((idx)/self.num_samples_per_slice) + 2
        overlay =  idx % self.num_samples_per_slice
        z_index = int(overlay/(int(self.n/self.stride) - 2))
        x_index = overlay % (int(self.n/self.stride) - 2)


        with h5py.File(self.file_path, 'r') as h5f:
            volume = h5f['Volume']
            sample = volume[x_index*self.stride: x_index*(self.stride+1), 
                            y_index-2:y_index+3, 
                            z_index*self.stride: z_index*(self.stride+1)]

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataloader(batch_size, number_of_gpus, stride, file_paths): 

    train_loader = DataLoader(
                ConcatDataset([VolumeDataset(path, stride) for path in file_paths]),
                batch_size=batch_size,
                shuffle=True,
                num_workers=number_of_gpus,
                pin_memory=True, # loads them directly in cuda pinned memory 
                drop_last=True) # drop the last incomplete batch

    return train_loader

