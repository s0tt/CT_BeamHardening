import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.functional import Tensor


def make_grid(data, numrows):
    """ Create numerical grid of input data which can be used to visualize 
    multi-dimensional data
    taken from & credits to: https://learnopencv.com/tensorboard-with-pytorch-lightning/

    Args:
        data: input data
        numrows: number of grid rows

    Returns:
        c: data grid
    """
    outer = (torch.Tensor.cpu(data).detach())
    b = np.array([]).reshape(0, outer.shape[2])  # column array
    c = np.array([]).reshape(numrows*outer.shape[2], 0)  # row array
    i = 0
    j = 0
    while(i < outer.shape[0]):
        img = outer[i]
        b = np.concatenate((img, b), axis=0)  # append new row to b
        j += 1
        if(j == numrows):
            c = np.concatenate((c, b), axis=1)  # apend new column b to c
            b = np.array([]).reshape(0, outer.shape[2])  # reinit b
            j = 0

        i += 1

    # if not enough rows return intermediate b array
    if not c.any():
        return b
    return c


def plot_pred_gt(x, pred, gt):
    fig = plt.figure(figsize=(16, 18))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                     nrows_ncols=(2, 3),
                     axes_pad=(0.15, 0.3),
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="edge",
                     cbar_size="7%",
                     cbar_pad=0.15,
                     cbar_set_cax=True,
                     )

    max_first_row = max([x.max(), gt.max(), pred.max()])
    min_first_row = min([x.min(), gt.min(), pred.min()])
    max_second_row = max([(x-gt).max(), (pred-gt).max(), (pred-x).max()])
    min_second_row = min([(x-gt).min(), (pred-gt).min(), (pred-x).min()])

    grid[0].set_title("Input (Poly)")
    im = grid[0].imshow(np.squeeze(np.array(torch.Tensor.cpu(
        x).detach().numpy())), vmin=min_first_row, vmax=max_first_row, cmap="gray")
    grid[1].set_title("Prediction (Poly-residual)")
    im = grid[1].imshow(np.squeeze(np.array(torch.Tensor.cpu(
        pred).detach().numpy())), vmin=min_first_row, vmax=max_first_row, cmap="gray")
    im = grid[2].imshow(np.squeeze(np.array(torch.Tensor.cpu(
        gt).detach().numpy())), vmin=min_first_row, vmax=max_first_row, cmap="gray")
    grid[2].set_title("Ground Truth (Mono)")
    grid[2].cax.colorbar(im)
    grid[2].cax.toggle_label(True)

    # Diff Poly Gt
    im = grid[3].imshow(np.squeeze(np.array(torch.Tensor.cpu(
        x-gt).detach().numpy())), vmin=min_second_row, vmax=max_second_row, cmap="gray")
    grid[3].set_title("Input (Poly) - Ground Truth (Mono)")
    # Diff Prediction Gt
    im = grid[4].imshow(np.squeeze(np.array(torch.Tensor.cpu(
        pred-gt).detach().numpy())), vmin=min_second_row, vmax=max_second_row, cmap="gray")
    grid[4].set_title("Prediction - Ground Truth (Mono)")
    # Diff Prediction Poly
    im = grid[5].imshow(np.squeeze(np.array(torch.Tensor.cpu(
        pred-x).detach().numpy())),  vmin=min_second_row, vmax=max_second_row, cmap="gray")
    grid[5].set_title("Prediction - Input (Poly)")
    grid[5].cax.colorbar(im)
    grid[5].cax.toggle_label(True)

    return fig


def plot_ct(data, clim=None):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(torch.Tensor.cpu(data).detach().numpy(), cmap="gray")
    if clim is not None:
        plt.clim(clim[0], clim[1])
    plt.colorbar()
    return fig
