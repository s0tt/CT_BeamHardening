import h5py
import numpy as np
import argparse

DATATYPE_USED = "float32"

def remove_slices_with_to_much_air(path_in: str): 
    with h5py.File(path_in, "r+") as f_in:

        def cut_volume_in_certain_dimension(mean_grey_value, dimension): 
            # dimension is either 0, 1 or 2 
            vol_hdf5 = f_in["Volume"]
            first_relevant_slice = False
            new_volume_dataset = None
            for slice in range(vol_hdf5.shape[dimension]):

                if dimension == 0: 
                    vol_data = vol_hdf5[slice, :, :]
                elif dimension == 1: 
                    vol_data = vol_hdf5[:, slice, :]
                elif dimension == 2: 
                    vol_data = vol_hdf5[:, :, slice]

                mean_grey_value_slice = (vol_data.flatten().sum())/vol_data.size

                if not first_relevant_slice and (mean_grey_value > mean_grey_value_slice):
                    pass
                else: 
                    if not first_relevant_slice:
                        first_relevant_slice = True 
                        subtract = slice
                    vol_data_3d = np.expand_dims(vol_data, axis=dimension)
                    if new_volume_dataset is None:
                        new_volume_dataset = f_in.create_dataset("NewVolume",
                            data=vol_data_3d, dtype=DATATYPE_USED, chunks=True,
                            maxshape=(None, None, None))
                    else:
                        new_volume_dataset.resize(new_volume_dataset.shape[dimension] +
                            vol_data_3d.shape[dimension], axis=dimension)
                        new_volume_dataset[slice-subtract, :, :] = vol_data_3d
            
            # removes end slices
            for slice in range(-1, -new_volume_dataset.shape[dimension], -1):
                if dimension == 0: 
                    vol_data = new_volume_dataset[slice, :, :]
                elif dimension == 1: 
                    vol_data = new_volume_dataset[:, slice, :]
                elif dimension == 2: 
                    vol_data = new_volume_dataset[:, :, slice]
                mean_slice_grey_value = vol_data.flatten().sum()
                if mean_slice_grey_value < mean_grey_value: 
                    new_volume_dataset.resize(new_volume_dataset.shape[dimension]-1, axis=dimension)
                else: 
                    break 
        
            new_volume_dataset.attrs['MATLAB_class'] = 'double'

            # renaming in h5py
            del f_in['Volume']
            f_in['Volume'] = f_in["NewVolume"]
            del f_in["NewVolume"]

        mean_grey_value = f_in["Volume"].attrs['Mean_grey_value']
        
        cut_volume_in_certain_dimension(mean_grey_value, 0)
        print("Dimension 1 done")
        cut_volume_in_certain_dimension(mean_grey_value, 1)
        print("Dimension 2 done")
        cut_volume_in_certain_dimension(mean_grey_value, 2)
        print("Dimension 3 done")



        """
        vol_hdf5 = f_in["Volume"]
        first_relevant_y_slice = False

        # Volume iteration loop (y slices are now in first dimension (after transpose))
        # removes the beginning slices
        for y_slice in range(vol_hdf5.shape[0]):
            vol_data = vol_hdf5[y_slice, :, :]
            mean_grey_value_slice = (vol_data.flatten().sum())/vol_data.size

            if not first_relevant_y_slice and (mean_grey_value > mean_grey_value_slice):
                pass
            else: 
                if not first_relevant_y_slice:
                    first_relevant_y_slice = True 
                    y_subtract = y_slice
                vol_data_3d = np.expand_dims(vol_data, axis=0)
                if new_volume_dataset_y is None:
                    new_volume_dataset_y = f_in.create_dataset("VolumeFirstDimCutted",
                        data=vol_data_3d, dtype=DATATYPE_USED, chunks=True,
                        maxshape=(None, None, None))
                else:
                    new_volume_dataset_y.resize(new_volume_dataset_y.shape[0] +
                        vol_data_3d.shape[0], axis=0)
                    new_volume_dataset_y[y_slice-y_subtract, :, :] = vol_data_3d
        
        # removes end slices
        for y_slice in range(-1, -new_volume_dataset_y.shape[0], -1):
            vol_data = new_volume_dataset_y[y_slice, :, :]
            mean_slice_grey_value = vol_data.flatten().sum()
            if mean_slice_grey_value < mean_grey_value: 
                new_volume_dataset_y.resize(new_volume_dataset_y.shape[0]-1, axis=0)
            else: 
                break 
        
        new_volume_dataset_y.attrs['MATLAB_class'] = 'double'


        # renaming in h5py
        del f_in['Volume']
        f_in['Volume'] = f_in["VolumeFirstDimCutted"]
        del f_in["VolumeFirstDimCutted"]
        
    
        # cut in second dimension
        vol_hdf5 = f_in["Volume"]
        first_relevant_x_slice = False

        # Volume iteration loop (y slices are now in first dimension (after transpose))
        # removes the beginning slices
        for x_slice in range(vol_hdf5.shape[1]):
            vol_data = vol_hdf5[:, x_slice, :]
            mean_grey_value_slice = (vol_data.flatten().sum())/vol_data.size

            if not first_relevant_x_slice and (mean_grey_value > mean_grey_value_slice):
                pass
            else: 
                if not first_relevant_x_slice: 
                    x_subtract = x_slice
                vol_data_3d = np.expand_dims(vol_data, axis=1)
                if new_volume_dataset_x is None:
                    new_volume_dataset_x = f_in.create_dataset("VolumeSecondDimCutted",
                        data=vol_data_3d, dtype=DATATYPE_USED, chunks=True,
                        maxshape=(None, None, None))
                else:
                    new_volume_dataset_x.resize(new_volume_dataset_x.shape[1] +
                        vol_data_3d.shape[1], axis=1)
                    new_volume_dataset_x[:, x_slice-x_subtract, :] = vol_data_3d
        
        # removes end slices
        for x_slice in range(-1, -new_volume_dataset_x.shape[1], -1):
            vol_data = new_volume_dataset_x[:, x_slice, :]
            mean_slice_grey_value = vol_data.flatten().sum()
            if mean_slice_grey_value < mean_grey_value: 
                new_volume_dataset_x.resize(new_volume_dataset_x.shape[1]-1, axis=1)
            else: 
                break 
        
        new_volume_dataset_x.attrs['MATLAB_class'] = 'double'


        # renaming in h5py
        del f_in['Volume']
        f_in['Volume'] = f_in["VolumeSecondDimCutted"]
        del f_in["VolumeSecondDimCutted"]


        # cut in second dimension
        vol_hdf5 = f_in["Volume"]
        first_relevant_z_slice = False

        # Volume iteration loop (y slices are now in first dimension (after transpose))
        # removes the beginning slices
        for z_slice in range(vol_hdf5.shape[2]):
            vol_data = vol_hdf5[:, :, z_slice]
            mean_grey_value_slice = (vol_data.flatten().sum())/vol_data.size

            if not first_relevant_z_slice and (mean_grey_value > mean_grey_value_slice):
                pass
            else: 
                if not first_relevant_z_slice: 
                    first_relevant_z_slice = True 
                    z_subtract = z_slice
                vol_data_3d = np.expand_dims(vol_data, axis=2)
                if new_volume_dataset_z is None:
                    new_volume_dataset_z = f_in.create_dataset("VolumeThirdDimCutted",
                        data=vol_data_3d, dtype=DATATYPE_USED, chunks=True,
                        maxshape=(None, None, None))
                else:
                    new_volume_dataset_z.resize(new_volume_dataset_z.shape[2] +
                        vol_data_3d.shape[2], axis=2)
                    new_volume_dataset_z[:, :z_slice-z_subtract] = vol_data_3d
        
        # removes end slices
        for z_slice in range(-1, -new_volume_dataset_z.shape[2], -1):
            vol_data = new_volume_dataset_z[:, :, z_slice]
            mean_slice_grey_value = vol_data.flatten().sum()
            if mean_slice_grey_value < mean_grey_value: 
                new_volume_dataset_z.resize(new_volume_dataset_z.shape[2]-1, axis=1)
            else: 
                break 
        
        new_volume_dataset_z.attrs['MATLAB_class'] = 'double'


        # renaming in h5py
        del f_in['Volume']
        f_in['Volume'] = f_in["VolumeThirdDimCutted"]
        del f_in["VolumeThirdDimCutted"]
        """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path-in", "-f", required=True, type=str,
                        help="absolut path of input hdf5 file")


    args = parser.parse_args()
    remove_slices_with_to_much_air(args.file_path_in)


if __name__ == "__main__":
    main()
