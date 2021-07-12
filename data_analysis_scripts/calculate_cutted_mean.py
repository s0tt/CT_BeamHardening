import h5py
import argparse

def add_cutted_mean_grey_value(path_in: str):
    with h5py.File(path_in, "r+") as f_in:

        vol_hdf5 = f_in["Volume"]

        # Volume iteration loop
        grey_value_sum = 0
        for y_slice in range(vol_hdf5.shape[1]):
            vol_slice = vol_hdf5[:, y_slice, :]
            grey_value_sum += vol_slice.flatten().sum()
            print(grey_value_sum)
        f_in.attrs['Mean_grey_value_cutted'] = grey_value_sum/vol_hdf5.size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path-in", "-f", required=True, type=str,
                        help="absolut path of input hdf5 file")
    args = parser.parse_args()
    add_cutted_mean_grey_value(args.file_path_in)


if __name__ == "__main__":
    main()
