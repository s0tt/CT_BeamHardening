import h5py
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../models'))

from utils import parse_dataset_paths


def add_cutted_mean_grey_value(path_in: str):
    with h5py.File(path_in, "r+") as f_in:

        vol_hdf5 = f_in["Volume"]

        # Volume iteration loop
        grey_value_sum = 0
        for y_slice in range(vol_hdf5.shape[1]):
            vol_slice = vol_hdf5[:, y_slice, :]
            grey_value_sum += vol_slice.flatten().sum()
            print(grey_value_sum)
        
        # add cutted mean as MetaData
        vol_hdf5.attrs['Mean_grey_value_cutted_before_rescaling'] = grey_value_sum/vol_hdf5.size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", "-f", required=True,
                        help="Path to json file that contains all datasets")
    parser.add_argument("--dataset-names", "-dn", required=False, nargs='+', default=["all"],
                        help="Names of the datasets of --file-in that should analysed")
    args = parser.parse_args()

    dataset_paths = parse_dataset_paths(args.file_in, args.dataset_names)

    for paths in dataset_paths:
        add_cutted_mean_grey_value(paths[0])
        add_cutted_mean_grey_value(paths[1])


if __name__ == "__main__":
    main()
