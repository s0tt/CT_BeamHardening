import unittest
import numpy as np 
import sys
import os
import h5py 

"""
    The testsuit must be executed on the ipvs-servers (volume data is stored there) 
    This test case is written in a brute force manner, because the algorithm 
    which is tested should not be tested with itself ... 
"""

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')) 

from dataloader import get_dataloader

volume_path_20_20_20 = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', "test_data/volume.hdf5")
volume_path_zuendkerze = "/net/pasnas01.informatik.uni-stuttgart.de/pool1/enpro-2021-voxie/reconstructed_volumes/zuendkerze_bad_reconstruction/volume.hdf5"


class TestDataloader_x20_y20_z20(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDataloader_x20_y20_z20, self).__init__(*args, **kwargs)
        number_of_gpus = 0 
        batch_size = 1
        self.stride = 3 
        self.num_pixel = 5 
        self.dataset_path_x20_y20_z20 = [(volume_path_20_20_20, volume_path_20_20_20)]    
        self.dataloader_x20_y20_z20 = get_dataloader(batch_size, number_of_gpus, self.num_pixel, self.stride, self.dataset_path_x20_y20_z20, shuffle=False)

    def test_x20_y20_z20(self):
        for idx, item in enumerate(self.dataloader_x20_y20_z20): 
            gt = item[1]
            if idx == 0: 
                with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[: self.num_pixel, :5, : self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

            if idx == 1: 
                with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[ self.stride: self.stride + self.num_pixel, :5, : self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

            if idx == 40: 
                with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[self.stride*4: self.stride*4 + self.num_pixel, 1:6, : self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

            if idx == 80: 
                with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[self.stride*2: self.stride*2 + self.num_pixel, 2:7, self.stride : self.stride + self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

            if idx == 140: 
                with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[self.stride*2: self.stride*2 + self.num_pixel, 3:8, self.stride*5 : self.stride*5 + self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

            if idx == 380: 
                with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[self.stride*2: self.stride*2 + self.num_pixel, 10:15, self.stride*3 : self.stride*3 + self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

            
            if idx == 527: 
                with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[self.stride*5: self.stride*5 + self.num_pixel, 14:19, self.stride*3 : self.stride*3 + self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")                                          


class TestDataloader_zuendkerze_bad(unittest.TestCase):
    # Volume shape: (3158x2304x3158)

    def __init__(self, *args, **kwargs):
        super(TestDataloader_zuendkerze_bad, self).__init__(*args, **kwargs)
        number_of_gpus = 0 
        batch_size = 1
        self.stride = 128 
        self.num_pixel = 256
        # 23 samples in x/ z slice 
        # --> 23* 23 = 529 samples per slice
        # --> 529*2300 = 1216700 slices
        self.dataset_path_zuendkerze = [(volume_path_zuendkerze,
                             volume_path_zuendkerze)]    
        self.dataloader_zuendkerze = get_dataloader(batch_size, number_of_gpus, self.num_pixel, self.stride, self.dataset_path_zuendkerze, shuffle=False)

    def test_zuendkerze(self):
        for idx, item in enumerate(self.dataloader_zuendkerze): 
            gt = item[1]
            if idx == 0: 
                with h5py.File(self.dataset_path_zuendkerze[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[: self.num_pixel, :5, : self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

            if idx == 99: 
                with h5py.File(self.dataset_path_zuendkerze[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[self.stride*7: self.stride*7 + self.num_pixel, :5, self.stride*4: self.stride*4 + self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

            if idx == 1999: 
                with h5py.File(self.dataset_path_zuendkerze[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[self.stride*19: self.stride*19 + self.num_pixel, 3:8, self.stride*1: self.stride*1 + self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

            if idx == 99999: 
                with h5py.File(self.dataset_path_zuendkerze[0][0], 'r') as h5f:
                    volume_bh = h5f['Volume']
                    sample_bh = np.expand_dims(volume_bh[self.stride*18: self.stride*18 + self.num_pixel, 189:194,  : self.num_pixel], axis=0)
                self.assertTrue(np.array_equal(sample_bh, gt), "Slices are not equal")

if __name__ == '__main__':
    unittest.main()