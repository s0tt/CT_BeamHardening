import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_sample(row1, row2, title):
    for j in range(row1.shape[0]):
        fig = plt.figure(figsize=(16, 18))
        plt.title(title + "Dim: " + str(j))
        n_cols = 5

        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(2, n_cols),
                        axes_pad=(0.15, 0.3),
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="edge",
                        cbar_size="7%",
                        cbar_pad=0.15,
                        cbar_set_cax=True,
                        )

        max_first_row = torch.max(row1[j]).detach().numpy()
        min_first_row = torch.min(row1[j]).detach().numpy()

        max_second_row = torch.max(row2[j]).detach().numpy()
        min_second_row = torch.min(row2[j]).detach().numpy()

        for i in range(row1.shape[1]):
            grid[i].set_title("Slice " + str(i))
            im = grid[i].imshow(np.squeeze(np.array(torch.Tensor.cpu(
                row1[j,i, :, :]).detach().numpy())), vmin=min_first_row, vmax=max_first_row, cmap="gray")

            grid[i+n_cols].set_title("Slice " + str(i))
            im2 = grid[i+n_cols].imshow(np.squeeze(np.array(torch.Tensor.cpu(
                row2[j,i, :, :]).detach().numpy())), vmin=min_second_row, vmax=max_second_row, cmap="gray")

            if i == n_cols-1:
                grid[n_cols-1].cax.colorbar(im)
                grid[2*n_cols-1].cax.colorbar(im2)
            # grid[2].cax.toggle_label(True)

# load data
al_path_in = "/home/so/enpro-2021-voxie/reconstructed_volumes/grating_holder/poly/volume_trans_cut_fc0_5_downsampled.hdf5"
cu_path_in = "/home/so/enpro-2021-voxie/reconstructed_volumes/schraube/poly/volume_transposed_and_cutted_fc0_5_downsampled.hdf5"

with h5py.File(al_path_in, "r") as f_in:
    vol_hdf5 = f_in["Volume"]
    al_sample = torch.from_numpy(np.array(vol_hdf5[15:20, :, :])).unsqueeze(0)
with h5py.File(cu_path_in, "r") as f_in:
    vol_hdf5 = f_in["Volume"]
    cu_sample = torch.from_numpy(np.array(vol_hdf5[15:20, :, :])).unsqueeze(0)


#plt.show()

batch = torch.stack((cu_sample[0,:,0:70, 0:70], al_sample[0,:, 0:70, 180:250]))
batch_transposed = batch.transpose(0, 3)

# With Learnable Parameters
m_al = nn.LayerNorm(al_sample.size()[1:], elementwise_affine=False)
m_cu = nn.LayerNorm(cu_sample.size()[1:], elementwise_affine=False)

m_batch = nn.LayerNorm(batch.size()[1:], elementwise_affine=False)
m_batch_transposed = nn.LayerNorm(batch_transposed.size()[1:])
# Without Learnable Parameters
# m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
al_sample_normed = m_al(al_sample)
cu_sample_normed = m_cu(cu_sample)
batch_normed = m_batch(batch)
batch_transposed_normed = m_batch_transposed(batch_transposed)

mean = torch.mean(batch, [1, 2, 3])
std = torch.std(batch, [1, 2, 3], unbiased=False)

batch_manual_normed = torch.div(torch.sub(batch, mean[:,None,None,None]), std[:,None,None, None])
batch_manual_reconstruct = torch.add(torch.multiply(batch_manual_normed, std[:,None,None,None]), mean[:,None,None, None])

m_batch_2d = nn.BatchNorm2d(5, affine=False)
batch_normed_2d = torch.zeros_like(batch)
for i in range(batch.shape[0]):
    batch_normed_2d[i,:,:,:] = m_batch_2d(batch[i,:,:,:].unsqueeze(0))



# Normalize over last two dimensions
# m = nn.LayerNorm([10, 10])
# # Normalize over last dimension of size 10
# m = nn.LayerNorm(10)
# Activating the module
# al_sample_normed = m(al_sample)
# cu_sample_normed = m(cu_sample)

#plot_sample(batch, batch_normed, "AL Sample normed")
#plot_sample(batch, batch_normed, "Batch")
plot_sample(batch, batch_manual_normed, "Batch")
plot_sample(batch, batch_manual_reconstruct, "Batch Reconstruct")
#plot_sample(batch, batch_normed_2d, "Batch")
#plot_sample(batch_transposed.transpose(0,3), batch_transposed_normed.transpose(0,3), "Batch Transposed")
# plot_sample(al_sample, al_sample_normed, "AL Sample normed")
# plot_sample(cu_sample, cu_sample_normed, "CU Sample normed")

plt.show()