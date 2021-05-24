import h5py
import numpy as np
import argparse
import time


def trans_hdf5_incremental(path_in: str, path_out):
    with h5py.File(path_in, "r") as f_in:
        if path_out is not None:
            name_new = path_out
        else:
            name_old = path_in.split(".hdf5")[0]
            name_new = name_old+"_transposed.hdf5"
        with h5py.File(name_new, "w") as f_out:
            new_volume_dataset = None

            # takeover all other existing groups beside "Volume"
            for key in f_in.keys():
                if key != "Volume":
                    # Get parent group name for copy
                    group_path = f_in[key].parent.name
                    # Check existence of group, else create group+parent
                    group_id = f_out.require_group(group_path)
                    f_in.copy(key, group_id, group_path+key)

            # Volume iteration loop
            vol_hdf5 = f_in["Volume"]
            for y_slice in range(vol_hdf5.shape[1]):
                vol_data = vol_hdf5[:, y_slice, :]
                vol_data_3d = np.expand_dims(vol_data, axis=0)
                if new_volume_dataset is None:
                    new_volume_dataset = f_out.create_dataset("Volume",
                        data=vol_data_3d, dtype='float64', chunks=True,
                        maxshape=(None, None, None))
                else:
                    new_volume_dataset.resize(new_volume_dataset.shape[0] +
                        vol_data_3d.shape[0], axis=0)
                    new_volume_dataset[y_slice, :, :] = vol_data_3d
            new_volume_dataset.attrs['MATLAB_class'] = 'double'
    return name_new


def compare_hdf5(f_hdf5_normal, f_hdf5_transposed, nr_slices=20):
    # open files and set cache to zero for better comparison
    with h5py.File(f_hdf5_normal, "r", rdcc_nbytes=0) as f_in: 
        with h5py.File(f_hdf5_transposed, "r", rdcc_nbytes=0) as f_out:
            vol_in = f_in["Volume"]
            vol_out = f_out["Volume"]
            in_times, out_times, diffs = [], [], []
            for y_slice in range(nr_slices):
                t_before_in = time.process_time()
                slice_in = vol_in[:, y_slice, :]
                t_after_in = time.process_time()
                t_before_out = time.process_time()
                slice_out = vol_out[y_slice, :, :]
                t_after_out = time.process_time()
                in_times.append(t_after_in - t_before_in)
                out_times.append(t_after_out - t_before_out)
                diffs.append((t_after_in - t_before_in) -
                    (t_after_out - t_before_out))
                print("Normal: {} \t Transposed: {} \tDiff: {}".format(
                        t_after_in - t_before_in, t_after_out-t_before_out,
                        (t_after_in - t_before_in) - (t_after_out-t_before_out)
                        ))
                print("Normal == Transposed? ", np.all(slice_in == slice_out))

            print("----------------Overall times ----------------\n \
                Normal: {} \t- Transposed: {} \t=Diff: {}".format(
                    np.mean(in_times), np.mean(out_times), np.mean(diffs)))
            print("Improvement (wrt to normal): {}%".format((np.mean(diffs)
                    / np.mean(in_times)) * 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path-in", "-f", required=True, type=str,
                        help="absolut path of input hdf5 file")

    parser.add_argument("--file-path-out", "-o", required=False, type=str, 
                        help="absolut path of output hdf5 file")

    parser.add_argument("--compare-nr-slices", "-c", required=False, type=int,
                        help="nr of y-slices to compare load times")

    args = parser.parse_args()
    f_out = trans_hdf5_incremental(args.file_path_in, args.file_path_out)

    if args.compare_nr_slices is not None:
        compare_hdf5(args.file_path_in, f_out, args.compare_nr_slices)


if __name__ == "__main__":
    main()
