import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing
import itertools
import h5py
import queue

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

        if idx == 20: 
            print("test")

        if self.transform:
            sample_gt = self.transform(sample_gt)
            sample_bh = self.transform(sample_bh)

        return [sample_bh, sample_gt]


def worker_fn(dataset, index_queue, output_queue):
   while True:
       try:
           index = index_queue.get(timeout=0)
       except multiprocessing.Queue.Empty:
           continue
       if index is None:
           break
       output_queue.put((index, dataset[index]))


# DataLoader as explained in: https://www.pytorchlightning.ai/blog/dataloaders-explained
class AsynchronLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, num_workers=1, prefetch_batches=2, **kwargs):

        super().__init__(dataset, batch_size, **kwargs)

        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.output_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=worker_fn, args=(self.dataset, index_queue, self.output_queue)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.prefetch()

    def prefetch(self):
        while (self.prefetch_index < len(self.dataset) and self.prefetch_index < self.index + 2 * self.num_workers * self.batch_size):
            # if the prefetch_index hasn't reached the end of the dataset
            # and it is not 2 batches ahead, add indexes to the index queues
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def get(self):
        self.prefetch()
        if self.index in self.cache:
            item = self.cache[self.index]
            del self.cache[self.index]
        else:
            while True:
                try:
                    (index, data) = self.output_queue.get(timeout=0)
                except queue.Empty:  # output queue empty, keep trying
                    continue
                if index == self.index:  # found our item, ready to return
                    item = data
                    break
                else:  # item isn't the one we want, cache for later
                    self.cache[index] = data

        self.index += 1
        return item

    def __iter__(self):
        self.index = 0
        self.cache = {}
        self.prefetch_index = 0
        self.prefetch()
        return self

def get_dataloader(batch_size, number_of_gpus, num_pixel, stride, volume_paths, shuffle=True):
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
                shuffle=shuffle,
                num_workers=number_of_gpus,
                pin_memory=True, # loads them directly in cuda pinned memory 
                drop_last=True) # drop the last incomplete batch
    
    return train_loader

