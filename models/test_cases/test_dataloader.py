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

from dataloader import ConcatDataset 
from dataloader import VolumeDatasetTrain 


volume_path_20_20_20 = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', "test_data/volume.hdf5")
volume_path_zuendkerze = "/net/pasnas01.informatik.uni-stuttgart.de/pool1/enpro-2021-voxie/reconstructed_volumes/zuendkerze_bad_reconstruction/volume_transposed.hdf5"


class TestDataloader_x20_y20_z20(unittest.TestCase):
    # volume shape: 20x20x20

    def __init__(self, *args, **kwargs):
        super(TestDataloader_x20_y20_z20, self).__init__(*args, **kwargs)
        self.stride = 3 
        self.num_pixel = 5 
        self.dataset_path_x20_y20_z20 = [(volume_path_20_20_20, volume_path_20_20_20)]    
        self.concatDataset_20_20_20 = ConcatDataset([VolumeDatasetTrain(path[0], path[1], self.num_pixel, self.stride) for path in self.dataset_path_x20_y20_z20])

    def test_idx_0(self): 
        dataset_bh = self.concatDataset_20_20_20[0][1]
        with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[ :5, : self.num_pixel, : self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")

    def test_idx_1(self): 
        dataset_bh = self.concatDataset_20_20_20[1][1]
        with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[:5, self.stride: self.stride + self.num_pixel, : self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")

    def test_idx_40(self): 
        dataset_bh = self.concatDataset_20_20_20[40][1]
        with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[1:6, self.stride*4: self.stride*4 + self.num_pixel, : self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")

    def test_idx_140(self): 
        dataset_bh = self.concatDataset_20_20_20[140][1]
        with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[3:8, self.stride*2: self.stride*2 + self.num_pixel, self.stride*5 : self.stride*5 + self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")

    def test_idx_380(self): 
        dataset_bh = self.concatDataset_20_20_20[380][1]
        with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[10:15, self.stride*2: self.stride*2 + self.num_pixel, self.stride*3 : self.stride*3 + self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")

    def test_idx_527(self): 
        dataset_bh = self.concatDataset_20_20_20[527][1]
        with h5py.File(self.dataset_path_x20_y20_z20[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[14:19, self.stride*5: self.stride*5 + self.num_pixel, self.stride*3 : self.stride*3 + self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")


class TestDataloader_zuendkerze_bad(unittest.TestCase):
    # Volume shape: (2304x3158x3158)

    def __init__(self, *args, **kwargs):
        super(TestDataloader_zuendkerze_bad, self).__init__(*args, **kwargs)
        self.stride = 128 
        self.num_pixel = 256
        # 23 samples in y/ z slice 
        # --> 23* 23 = 529 samples per slice
        # --> 529*2300 = 1216700 slices
        self.dataset_path_zuendkerze = [(volume_path_zuendkerze,
                             volume_path_zuendkerze)]    
        
        self.concatDataset_zuendkerze = ConcatDataset([VolumeDatasetTrain(path[0], path[1], self.num_pixel, self.stride) for path in self.dataset_path_zuendkerze])

    def test_idx_0(self): 
        dataset_bh = self.concatDataset_zuendkerze[0][1]
        with h5py.File(self.dataset_path_zuendkerze[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[:5, : self.num_pixel, : self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")

    def test_idx_99(self): 
        dataset_bh = self.concatDataset_zuendkerze[99][1]
        with h5py.File(self.dataset_path_zuendkerze[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[ :5, self.stride*7: self.stride*7 + self.num_pixel, self.stride*4: self.stride*4 + self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")

    def test_idx_1999(self): 
        dataset_bh = self.concatDataset_zuendkerze[1999][1]
        with h5py.File(self.dataset_path_zuendkerze[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[3:8, self.stride*21: self.stride*21 + self.num_pixel, self.stride*17: self.stride*17 + self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")

    def test_idx_99999(self): 
        dataset_bh = self.concatDataset_zuendkerze[99999][1]
        with h5py.File(self.dataset_path_zuendkerze[0][0], 'r') as h5f:
            volume_bh = h5f['Volume']
            sample_bh_self_extracted = volume_bh[189:194, self.stride*18: self.stride*18 + self.num_pixel,  : self.num_pixel]
        self.assertTrue(np.array_equal(sample_bh_self_extracted, dataset_bh), "Slices are not equal")

if __name__ == '__main__':
    unittest.main()