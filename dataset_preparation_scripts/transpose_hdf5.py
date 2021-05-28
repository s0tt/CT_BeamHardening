import h5py
import numpy as np
import argparse
import time

DATATYPE_USED = "float32"

def trans_hdf5_incremental(path_in: str, path_out: str):
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
            
            vol_hdf5 = f_in["Volume"]
            # Volume iteration loop
            grey_value_sum = 0
            for y_slice in range(vol_hdf5.shape[1]):
                vol_data = vol_hdf5[:, y_slice, :]
                grey_value_sum += vol_data.flatten().sum()

                vol_data_3d = np.expand_dims(vol_data, axis=0)
                if new_volume_dataset is None:
                    new_volume_dataset = f_out.create_dataset("Volume",
                        data=vol_data_3d, dtype=DATATYPE_USED, chunks=True,
                        maxshape=(None, None, None))
                else:
                    new_volume_dataset.resize(new_volume_dataset.shape[0] +
                        vol_data_3d.shape[0], axis=0)
                    new_volume_dataset[y_slice, :, :] = vol_data_3d
            
            new_volume_dataset.attrs['MATLAB_class'] = 'double'
            new_volume_dataset.attrs['Mean_grey_value'] = grey_value_sum/vol_hdf5.size

def remove_slices_with_to_much_air(path_in: str): 
    with h5py.File(path_in, "r") as f_in:
        if path_out is not None:
            name_new = path_out
        else:
            name_old = path_in.split(".hdf5")[0]
            name_new = name_old+"_transposed.hdf5"
        with h5py.File(name_new, "w") as f_out:
            new_volume_dataset = None
            image_dataset.resize(image_dataset.shape[0]+img_current.shape[0], axis=0)
            
            # takeover all other existing groups beside "Volume"
            for key in f_in.keys():
                if key != "Volume":
                    # Get parent group name for copy
                    group_path = f_in[key].parent.name
                    # Check existence of group, else create group+parent
                    group_id = f_out.require_group(group_path)
                    f_in.copy(key, group_id, group_path+key)
            
            vol_hdf5 = f_in["Volume"]
            # Volume iteration loop
            grey_value_sum = 0
            for y_slice in range(vol_hdf5.shape[1]):
                vol_data = vol_hdf5[:, y_slice, :]
                grey_value_sum += vol_data.flatten().sum()

                vol_data_3d = np.expand_dims(vol_data, axis=0)
                if new_volume_dataset is None:
                    new_volume_dataset = f_out.create_dataset("Volume",
                        data=vol_data_3d, dtype=DATATYPE_USED, chunks=True,
                        maxshape=(None, None, None))
                else:
                    new_volume_dataset.resize(new_volume_dataset.shape[0] +
                        vol_data_3d.shape[0], axis=0)
                    new_volume_dataset[y_slice, :, :] = vol_data_3d
            
            new_volume_dataset.attrs['MATLAB_class'] = 'double'
            new_volume_dataset.attrs['Mean_grey_value'] = grey_value_sum/vol_hdf5.size

def compare_hdf5(f_hdf5_normal, f_hdf5_transposed, nr_slices=20):
    # open files and set cache to zero for better comparison
    t_before_open_in = time.process_time()
    with h5py.File(f_hdf5_normal, "r", rdcc_nbytes=0) as f_in:
        t_after_open_in = time.process_time()
        with h5py.File(f_hdf5_transposed, "r", rdcc_nbytes=0) as f_out:
            t_after_open_out = time.process_time()
            vol_in = f_in["Volume"]
            t_after_vol_in = time.process_time()
            vol_out = f_out["Volume"]
            t_after_vol_out = time.process_time()
            in_times, out_times, diffs = [], [], []
            print("File open\t| Normal: {} \t Transposed: {} \tDiff: {}".format(
                t_after_open_in - t_before_open_in, t_after_open_out - t_after_open_in,
                (t_after_open_in - t_before_open_in) - (t_after_open_out - t_after_open_in)
                ))
            print("Volume access\t| Normal: {} \t Transposed: {} \tDiff: {}\n".format(
                t_after_vol_in - t_after_open_out, t_after_vol_out - t_after_vol_in,
                (t_after_vol_in - t_after_open_out) - (t_after_vol_out - t_after_vol_in)
                ))
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
                print("Slice {}\t| Normal: {} \t Transposed: {} \tDiff: {}".format(
                        y_slice,
                        t_after_in - t_before_in, t_after_out-t_before_out,
                        (t_after_in - t_before_in) - (t_after_out-t_before_out)
                        ))
                print("Normal == Transposed? ", np.all(slice_in == slice_out))

            print("----------------Overall times ----------------\n \
                Normal: {} \t- Transposed: {} \t=Diff: {}".format(
                    np.mean(in_times), np.mean(out_times), np.mean(diffs)))
            print("Improvement (wrt to normal): {}%".format((np.mean(diffs)
                    / np.mean(in_times)) * 100))
    return np.array([np.mean(in_times), np.mean(out_times), np.mean(diffs)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path-in", "-f", required=True, type=str,
                        help="absolut path of input hdf5 file")

    parser.add_argument("--file-path-out", "-o", required=False, type=str, 
                        help="absolut path of output hdf5 file")

    parser.add_argument("--file-path-compare", "-fc", required=False, type=str, 
                        help="absolut path of hdf5 file for comparison")

    parser.add_argument("--compare-nr-slices", "-s", required=False, type=int,
                        help="nr of y-slices to compare load times")
    parser.add_argument("--compare-cycles", "-c", required=False, type=int,
                        help="nr of cycles")

    args = parser.parse_args()
    if args.file_path_compare is None:
        trans_hdf5_incremental(args.file_path_in, args.file_path_out)

    elif args.compare_cycles is not None:
        if args.compare_nr_slices is None:
            args.compare_nr_slices = 20

        overall_data = np.array([0, 0, 0], dtype=float)
        for cyc in range(args.compare_cycles):
            res = compare_hdf5(args.file_path_in, args.file_path_compare,
                                args.compare_nr_slices)
            overall_data += (1/int(args.compare_cycles)) * res

        print("----------------Overall over all cycles ----------------\n \
                Normal: {} \t- Transposed: {} \t=Diff: {}".format(
                overall_data[0], overall_data[1], overall_data[2]))               

        print("Improvement (wrt to normal): {}%".format((overall_data[2]
                / overall_data[0]) * 100))


if __name__ == "__main__":
    main()
