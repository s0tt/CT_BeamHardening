
from sys import maxsize
import h5py
import numpy as np
import nexusformat.nexus as nx # at the moment just needed for debugging purposes
import time
import struct
from dataloader import get_dataloader
from visualization import make_grid
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
from mpl_toolkits.axes_grid1 import ImageGrid

detector_pixel_size = 0.000127 # [m]
distance_source_detector = 1.229268# [m]
# your code here    
def trans_hdf5_incremental(f_hdf5):
  with h5py.File(filename, "r") as f_in:
    new_string = f_hdf5.split(".hdf5")[0]
    with h5py.File(new_string+"_transposed_float32.hdf5", "w") as f_out:
      for key in f_in.keys():
        if key != "Volume":
          # Get parent group name for copy
          group_path = f_in[key].parent.name
          # Check existence of group, else create group incuding parent
          group_id = f_out.require_group(group_path)
          f_in.copy(key, group_id, group_path+key)

      new_volume_dataset = None
      vol_hdf5 = f_in["Volume"]
      for y_slice in range(20):
        vol_data = vol_hdf5[:,y_slice,:]
        vol_data_3d = np.expand_dims(vol_data, axis=0)
        if new_volume_dataset is None:
          new_volume_dataset = f_out.create_dataset("Volume", data=vol_data_3d, dtype='float32', maxshape=(20,20,20))
        else:
          new_volume_dataset.resize(new_volume_dataset.shape[0]+vol_data_3d.shape[0], axis=0)
          new_volume_dataset[y_slice, :, :] = vol_data_3d
      new_volume_dataset.attrs['MATLAB_class'] = 'double'

def compare_hdf5(f_hdf5_normal,f_hdf5_transposed, nr_slices=20):
  #open files and set cache to zero for better comparison
  with h5py.File(f_hdf5_normal, "r", rdcc_nbytes=0) as f_in: 
    with h5py.File(f_hdf5_transposed, "r", rdcc_nbytes=0) as f_out:

      vol_in = f_in["Volume"]
      vol_out= f_out["Volume"]
      new_volume_dataset = None
      in_times, out_times, diffs = [],[],[]
      for y_slice in range(nr_slices):
        t_before_in = time.process_time()
        slice_in = vol_in[:,y_slice,:]
        t_after_in = time.process_time()
        t_before_out = time.process_time()
        slice_out = vol_out[y_slice,:,:]
        t_after_out = time.process_time()
        in_times.append(t_after_in- t_before_in)
        out_times.append(t_after_out-t_before_out)
        diffs.append((t_after_in- t_before_in)- (t_after_out-t_before_out))
        print("Normal: {} \t Transposed: {} \tDiff: {}".format(t_after_in- t_before_in, t_after_out-t_before_out, 
              (t_after_in- t_before_in)- (t_after_out-t_before_out)))
        print("Normal == Transposed? ", np.all(slice_in==slice_out))
      
      print("----------------Overall times ----------------\n Normal: {} \t Transposed: {} \tDiff: {}".format(np.mean(in_times), 
          np.mean(out_times), np.mean(diffs)))
      print("Improvement (wrt to normal): {}%".format((np.mean(diffs)/np.mean(in_times))* 100))

        

filename = "/home/so/Git/dl_beamhardening/test_data/volume.hdf5"
f = nx.nxload(filename)
print(f.tree)

def double_to_hex(f):
    return hex(struct.unpack('@Q', struct.pack('@d', f))[0])

type_data = ""
start = time.process_time()
trans_hdf5_incremental(filename)
compare_hdf5(filename)

with h5py.File(filename, "r") as f:
  print("File open:\t", time.process_time() - start)
  volume_dataset = f.get("Volume")
  print("get Volume:\t", time.process_time() - start)
  vol_data = np.zeros((volume_dataset.shape), dtype=np.float64)
  volume_dataset.read_direct(vol_data)
  print("read_direct volume:", time.process_time() - start)
  vol_transposed = np.transpose(vol_data, (0, 2, 1)) #correct transpose

  #type_dataset.read_direct(type_data)
  assert np.all(vol_transposed[:,:,0] == vol_data[:,0,:])
  assert np.all(vol_transposed[:,:,5] == vol_data[:,5,:])
  assert np.all(vol_transposed[:,:,15] == vol_data[:,15,:])
  

  f = open("trans_gt.txt", "w")
  com_data_list = []
  comp_data_array = np.zeros((20,20,20), dtype=np.float64)
  for y_slice in range(20):
    comp_data = vol_transposed[:,:,y_slice]
    #comp_data_array[y_slice, :, :] = comp_data
    #comp_data_array[y_slice, :, :] = vol_transposed[:][:][y_slice]
    
    comp_data_array[y_slice, :, :] = vol_data[:,y_slice,:]
    com_data_list.append(vol_data[:,y_slice,:])
    for val in np.nditer(comp_data, order="C"): #iterate in C style over array
        str_hex = double_to_hex(val)
        str_hex = str_hex.split("x")[1]
        str_new = ' '.join(b+a for a,b in zip(str_hex[::-2], str_hex[::-1][1::2]))
        f.write(str_new + " ")
  f.close()

  #np.array(comp_data_all).astype('float64').tofile("trans_gt_tofile_all")
  #f = np.save("trans_gt", np.reshape(comp_data,comp_data.shape, order="C"))
  com_data_list[0].astype('float64').tofile("comp_data_list_0")
  #com_data_list.astype('float64').tofile("comp_data_list_all")
  comp_data_array.astype('float64').tofile("comp_data_all")


  f2 = open("trans_gt_integers.txt", "w")
  for y_slice in range(20):
    comp_data2 = vol_transposed[:][:][y_slice]
    for idx, val in enumerate(np.nditer(comp_data2, order="C")):
      f2.write(str(val) +" ")
      if (idx+1)%4 == 0:
        f2.write("\n")
  f2.close()
  #f.close()



with h5py.File("slice_0to1.hdf5", "w") as f_transp:
  image_dataset = f_transp.create_dataset("Image", data=vol_data[:,0:2,:], dtype='float64', chunks=True)
  image_dataset.attrs['MATLAB_class'] = 'double'

with h5py.File("transposed_test.hdf5", "w") as f_transp:
  image_dataset = f_transp.create_dataset("Image", data=vol_transposed, dtype='float64', chunks=True)
  image_dataset.attrs['MATLAB_class'] = 'double'

with h5py.File("normal_test.hdf5", "w") as f_normal:
  image_dataset = f_normal.create_dataset("Image", data=vol_data, dtype='float64', chunks=True)
  image_dataset.attrs['MATLAB_class'] = 'double'

with h5py.File("normal_test_yiterate.hdf5", "w") as f_normal:
  image_dataset = f_normal.create_dataset("Image", data=vol_data[:][0:20][:], dtype='float64', chunks=True)
  image_dataset.attrs['MATLAB_class'] = 'double'

def plot_256slices_from_hdf5(clim, loader, indices, rows, fig_path):
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Volume iteration loop
    data_img = []
    data_gt = []
    cnt = 0
    for img, gt in iter(loader):
        if cnt % 5 == 0:
            print("Counter:", cnt)
        #data.append(img[:, 2, :, :])

        data_img.append(np.squeeze(np.array(img[:, 2, :, :])))
        data_gt.append(np.squeeze(np.array(gt[:, 2, :, :])))
        # Set up figure and image grid
        fig = plt.figure(figsize=(16, 9))

        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
        #plt.subplot(1,3,1)
        grid[0].set_title("Poly")
        im = grid[0].imshow(np.squeeze(np.array(img[:, 2, :, :])), cmap="gray")
        #im.clim(clim[0], clim[1])
        #plt.subplot(1,3,2)
        im = grid[1].imshow(np.squeeze(np.array(gt[:, 2, :, :])), cmap="gray")
        grid[1].set_title("Mono (Ground-Truth)")
        #im.clim(clim[0], clim[1])
        #plt.subplot(1,3,3)
        grid[1].cax.colorbar(im)
        grid[1].cax.toggle_label(True)
        
        #plt.colorbar()
        plt.savefig(fig_path+"img_"+str(cnt)+".png")
        plt.close()
        cnt += 1
    data_stack_img = np.stack(data_img)
    data_stack_gt = np.stack(data_gt)
    #print("Data stacked:", data_stack.shape)
    #print("Max:\t{} | Min:\t{}".format(np.amax(img), np.amin(img)))
    plt.figure(figsize=(12,8), dpi= 100)
    grid_array = make_grid(data_stack_img,int(np.sqrt(data_stack_img.shape[0])))
    plt.imshow(grid_array, cmap="gray")
    plt.clim(clim[0], clim[1])
    plt.colorbar()
    plt.savefig(fig_path+"grid_view_all_ct.png")

    plt.figure(figsize=(12,8), dpi= 100)
    grid_array = make_grid(data_stack_gt,int(np.sqrt(data_stack_gt.shape[0])))
    plt.imshow(grid_array, cmap="gray")
    plt.clim(clim[0], clim[1])
    plt.colorbar()
    plt.savefig(fig_path+"grid_view_all_gt.png")
            
        #plt.show()

from torch.utils.data.sampler import SequentialSampler
#path_in = "/net/pasnas01/pool1/enpro-2021-voxie/reconstructed_volumes/cable_holder_chain/poly/volume.hdf5"
path_ct = "/home/so/TIFFs/vol_cut_cableHolder_poly.hdf5"
path_gt = "/home/so/TIFFs/vol_cut_cableHolder_mono.hdf5"
path_out = "/home/so/TIFFs/cableBoth128/"

loader = get_dataloader(1, 2, 128, 128, [(path_ct, path_gt)], 
                                sampler=None, shuffle=False)
idx = list(range(len(loader.dataset)))
print("Idx:", idx)
sampler = SequentialSampler(idx)
loader = get_dataloader(1, 2, 128, 128, [(path_ct, path_gt)], 
                                sampler=sampler, shuffle=False)
print("Loader data length:", len(loader.dataset))
print("Loader datasets:", loader.dataset.len_datasets)
print("Num datasets:", loader.dataset.num_datasets)

plot_256slices_from_hdf5([-2, 100],loader, idx, 2, path_out)     