import numpy as np
import matplotlib.pyplot as plt
import torch

def make_grid(output,numrows):
    outer=(torch.Tensor.cpu(output).detach())
    plt.figure(figsize=(20,5))
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