import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.functional import Tensor

# taken from & credits to: https://learnopencv.com/tensorboard-with-pytorch-lightning/
def make_grid(output,numrows):
    outer=(torch.Tensor.cpu(output).detach())
    b=np.array([]).reshape(0,outer.shape[2])
    c=np.array([]).reshape(numrows*outer.shape[2],0)
    i=0
    j=0
    while(i < outer.shape[1]):
        img=outer[0][i]
        b=np.concatenate((img,b),axis=0)
        j+=1
        if(j==numrows):
            c=np.concatenate((c,b),axis=1)
            b=np.array([]).reshape(0,outer.shape[2])
            j=0
            
        i+=1
    return c

def plot_pred_gt(pred: Tensor, gt: Tensor):
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

    grid[0].set_title("Model prediction(residual)")
    im = grid[0].imshow(np.squeeze(np.array(pred.detach().numpy())), cmap="gray")
    im = grid[1].imshow(np.squeeze(np.array(gt.detach().numpy())), cmap="gray")
    grid[1].set_title("Mono (Ground-Truth)")
    grid[1].cax.colorbar(im)
    grid[1].cax.toggle_label(True)
    #plt.savefig(name+".png")
    return fig

def plot_ct(data: Tensor, clim=None):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(data.detach().numpy(), cmap="gray")
    if clim is not None:
        plt.clim(clim[0], clim[1])
    plt.colorbar()
    return fig