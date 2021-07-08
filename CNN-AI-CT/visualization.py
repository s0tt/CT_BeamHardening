import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.functional import Tensor

def make_grid(data,numrows):
    """ Create numerical grid of input data which can be used to visualize 
    multi-dimensional data
    taken from & credits to: https://learnopencv.com/tensorboard-with-pytorch-lightning/

    Args:
        data: input data
        numrows: number of grid rows

    Returns:
        c: data grid
    """
    outer=(torch.Tensor.cpu(data).detach())
    b = np.array([]).reshape(0,outer.shape[2]) # column array 
    c = np.array([]).reshape(numrows*outer.shape[2],0) # row array
    i = 0
    j = 0
    while(i < outer.shape[0]):
        img = outer[i]
        b = np.concatenate((img,b),axis=0) # append new row to b
        j += 1
        if(j == numrows):
            c = np.concatenate((c,b),axis=1) # apend new column b to c
            b = np.array([]).reshape(0,outer.shape[2]) # reinit b
            j = 0
            
        i+=1
    
    # if not enough rows return intermediate b array
    if not c.any():
        return b
    return c

def plot_pred_gt(x, pred, gt):
    fig = plt.figure(figsize=(16, 9))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                nrows_ncols=(1,3),
                axes_pad=0.15,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="7%",
                cbar_pad=0.15,
                )
    grid[0].set_title("Input (Poly)")
    im = grid[0].imshow(np.squeeze(np.array(torch.Tensor.cpu(x).detach().numpy())), cmap="gray")
    grid[1].set_title("Prediction (Poly-residual)")
    im = grid[1].imshow(np.squeeze(np.array(torch.Tensor.cpu(pred).detach().numpy())), cmap="gray")
    im = grid[2].imshow(np.squeeze(np.array(torch.Tensor.cpu(gt).detach().numpy())), cmap="gray")
    grid[2].set_title("Ground Truth (Mono)")
    grid[2].cax.colorbar(im)
    grid[2].cax.toggle_label(True)
    #plt.savefig(name+".png")
    return fig

def plot_ct(data, clim=None):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(torch.Tensor.cpu(data).detach().numpy(), cmap="gray")
    if clim is not None:
        plt.clim(clim[0], clim[1])
    plt.colorbar()
    return fig