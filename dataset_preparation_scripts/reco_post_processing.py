import h5py
import numpy as np
import argparse
import os

DATATYPE_USED = "float32"
gv_factor = 1 # scaling factor for the mean grey value


def trans_hdf5_incremental(path_in: str, path_out: str):
    with h5py.File(path_in, "r") as f_in:
        with h5py.File(path_out, "w") as f_out:

            vol_hdf5 = f_in["Volume"]
            new_volume_dataset = None 

            # Volume iteration loop
            grey_value_sum = 0
            for y_slice in range(vol_hdf5.shape[1]):
                vol_slice = vol_hdf5[:, y_slice, :]
                grey_value_sum += vol_slice.flatten().sum()

                if new_volume_dataset is None:
                    vol_slice = np.expand_dims(vol_slice, axis=0)
                    new_volume_dataset = f_out.create_dataset("Volume",
                    data=vol_slice, dtype=DATATYPE_USED, chunks=True,
                    maxshape=(None, None, None))
                else:
                    new_volume_dataset.resize(new_volume_dataset.shape[0] + 1, axis=0)
                    new_volume_dataset[y_slice, :, :] = vol_slice
            
            new_volume_dataset.attrs['MATLAB_class'] = 'double'
            new_volume_dataset.attrs['Mean_grey_value'] = grey_value_sum/vol_hdf5.size

def cut_volume_by_axis(axis: int, cuts_axis: tuple, path_old: str, path_new: str):
    """
        Cuts a volume based on the input of cuts_axis in the dimension of axis
        Params:
        cuts_axis: tuple, 
            where first entry is the index of the first relevant slide 
            <=> number of removed slides on low side and
            the second entry is the number of slides removed on the high side.
    """
    with h5py.File(path_old, "r+") as f_in: 
        with h5py.File(path_new, "w") as f_out:

            vol_hdf5 = f_in["Volume"]
            new_volume_dataset = None

            for slice_nr in range(cuts_axis[0], vol_hdf5.shape[axis]):
                if axis == 0: 
                    vol_slice = vol_hdf5[slice_nr, :, :]
                elif axis == 1: 
                    vol_slice = vol_hdf5[:, slice_nr, :]
                elif axis == 2: 
                    vol_slice = vol_hdf5[:, :, slice_nr]

                if new_volume_dataset is None:
                    vol_slice = np.expand_dims(vol_slice, axis=axis)
                    new_volume_dataset = f_out.create_dataset("Volume",
                    data=vol_slice, dtype=DATATYPE_USED, chunks=True,
                    maxshape=(None, None, None))
                else:
                    shape_list = list(new_volume_dataset.shape)
                    shape_list[axis] = shape_list[axis] + 1 
                    new_volume_dataset.resize(tuple(shape_list))

                    if axis == 0: 
                        new_volume_dataset[slice_nr-cuts_axis[0], :, :] = vol_slice
                    elif axis == 1: 
                        new_volume_dataset[:, slice_nr-cuts_axis[0], :] = vol_slice
                    elif axis == 2: 
                        new_volume_dataset[:, :, slice_nr-cuts_axis[0]] = vol_slice
            
            # removes end slices
            new_volume_dataset.resize(new_volume_dataset.shape[axis]-cuts_axis[1], axis=axis)
        
            new_volume_dataset.attrs['MATLAB_class'] = 'double'


def cut_air_slices_by_axis(axis: int, path_old: str, path_new:str) -> tuple:  
    """
        removes the edge slices in both directions perpendicular to the passed axis, 
        where the mean grey value is lower than the grey value of the full volume 
        multiplied with a scaling factor:  
            (mean_grey_value*gv_factor > mean_grey_value_slice)

        @return: 
            (slices_start, slices_end): 
                slices_start:  number of removed slices in the beginning 
                slices_end: number of removed slices in the end
    """

    if axis not in [0, 1, 2]: 
        raise Exception("Axis must be either 0, 1 or 2")

    with h5py.File(path_old, "r+") as f_in:
        mean_grey_value = f_in["Volume"].attrs['Mean_grey_value']
        with h5py.File(path_new, "w") as f_out:

            vol_hdf5 = f_in["Volume"]
            first_relevant_slice = False
            new_volume_dataset = None

            for slice_nr in range(vol_hdf5.shape[axis]):
                if axis == 0: 
                    vol_slice = vol_hdf5[slice_nr, :, :]
                elif axis == 1: 
                    vol_slice = vol_hdf5[:, slice_nr, :]
                elif axis == 2: 
                    vol_slice = vol_hdf5[:, :, slice_nr]

                mean_grey_value_slice = (vol_slice.flatten().sum())/vol_slice.size

                if not first_relevant_slice and (mean_grey_value*gv_factor > mean_grey_value_slice):
                    pass
                else: 
                    if not first_relevant_slice:
                        slices_start = slice_nr
                        first_relevant_slice = True 

                    if new_volume_dataset is None:
                        vol_slice = np.expand_dims(vol_slice, axis=axis)
                        new_volume_dataset = f_out.create_dataset("Volume",
                        data=vol_slice, dtype=DATATYPE_USED, chunks=True,
                        maxshape=(None, None, None))
                    else: 
                        shape_list = list(new_volume_dataset.shape)
                        shape_list[axis] = shape_list[axis] + 1 
                        new_volume_dataset.resize(tuple(shape_list))

                        if axis == 0: 
                            new_volume_dataset[slice_nr-slices_start, :, :] = vol_slice
                        elif axis == 1: 
                            new_volume_dataset[:, slice_nr-slices_start, :] = vol_slice
                        elif axis == 2: 
                            new_volume_dataset[:, :, slice_nr-slices_start] = vol_slice
                        
            
            # removes end slices
            for slice_nr in range(new_volume_dataset.shape[axis]):
                if axis == 0: 
                    vol_slice = new_volume_dataset[-1, :, :]
                elif axis == 1: 
                    vol_slice = new_volume_dataset[:, -1, :]
                elif axis == 2: 
                    vol_slice = new_volume_dataset[:, :, -1]
                mean_slice_grey_value = vol_slice.flatten().sum()/vol_slice.size
                if mean_slice_grey_value < mean_grey_value*gv_factor: 
                    new_volume_dataset.resize(new_volume_dataset.shape[axis]-1, axis=axis)
                else: 
                    slices_end = slice_nr
                    break 
        
            new_volume_dataset.attrs['MATLAB_class'] = 'double'
            new_volume_dataset.attrs['Mean_grey_value'] = mean_grey_value

    return (slices_start, slices_end)

def add_headers_and_metadata(path_old: str, path_new: str): 
    """
        Adds all headers and metadata from the old volume without the 
        volume dataset to the newly created volume. 
    """
    with h5py.File(path_old, "r") as f_in:
        with h5py.File(path_new, "r+") as f_out:
            for key in f_in.keys():
                if key != "Volume":
                    # Get parent group name for copy
                    group_path = f_in[key].parent.name
                    # Check existence of group, else create group+parent
                    group_id = f_out.require_group(group_path)
                    f_in.copy(key, group_id, group_path+key)
                              

def cut_volume(path_in_poly: str, path_in_mono: str):
    """
        Removes slices with to much air from all 
        6 edge planes of the cuboid. path_in <=> path_out 
        Cut is made for both volumes equivalend based on 
        the grey values of the poly volume
    """

    
    # Poly part 
    name_old_poly = path_in_poly.split(".hdf5")[0]
    name_new_0_poly = name_old_poly+"_dim_0_reduced.hdf5"
    name_new_1_poly = name_old_poly+"_dim_1_reduced.hdf5"

    cuts_x = cut_air_slices_by_axis(0, path_in_poly, name_new_0_poly)
    os.remove(path_in_poly)
    cuts_y = cut_air_slices_by_axis(1, name_new_0_poly, name_new_1_poly)
    os.remove(name_new_0_poly)
    cuts_z = cut_air_slices_by_axis(2, name_new_1_poly, path_in_poly)
    os.remove(name_new_1_poly)
    

    # Mono part 
    name_old_mono = path_in_mono.split(".hdf5")[0]
    name_new_0_mono = name_old_mono+"_dim_0_reduced.hdf5"
    name_new_1_mono = name_old_mono+"_dim_1_reduced.hdf5"
    
    cut_volume_by_axis(0, cuts_x, path_in_mono, name_new_0_mono)
    os.remove(path_in_mono)
    cut_volume_by_axis(1, cuts_y, name_new_0_mono, name_new_1_mono)
    os.remove(name_new_0_mono)
    cut_volume_by_axis(2, cuts_z, name_new_1_mono, path_in_mono)
    os.remove(name_new_1_mono)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path-in-mono", "-fm", required=True, type=str,
                        help="absolut path of input hdf5 file")

    parser.add_argument("--file-path-in-poly", "-fp", required=True, type=str,
                        help="absolut path of input hdf5 file")


    parser.add_argument("--file-path-out-mono", "-om", required=True, type=str, 
                        help="absolut path of output hdf5 file")
    
    parser.add_argument("--file-path-out-poly", "-op", required=True, type=str, 
                        help="absolut path of output hdf5 file")


    args = parser.parse_args()

    
    trans_hdf5_incremental(args.file_path_in_mono, args.file_path_out_mono)
    trans_hdf5_incremental(args.file_path_in_poly, args.file_path_out_poly)
    
    cut_volume(args.file_path_out_poly, args.file_path_out_mono)

    add_headers_and_metadata(args.file_path_in_mono, args.file_path_out_mono)
    add_headers_and_metadata(args.file_path_in_poly, args.file_path_out_poly)

  

if __name__ == "__main__":
    main()
