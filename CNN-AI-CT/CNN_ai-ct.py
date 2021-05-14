import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import numpy as np
from PIL import Image 

device = "cuda" if torch.cuda.is_available() else "cpu"
number_of_nodes = 1 
number_of_gpus = 0 
batch_size = 12


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class SliceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (string): Path to the volume data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_path = file_path
        self.transform = transform

    def __len__(self):
        """
        # load here maybe data and return the first dimension - 4 (the outer most four slices are not usefull)
        We do also have to take care for the patches of size 256 x 256 x 5

        """
        return len("test")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        """
            supply here the file reading and slice selection (Select always also the neighbouring 4 slices)
            The index 0 should correspond to the middle slice number 3 (counted from 1) and the highest index to the middle slice __len__ + 2.
            We do also have to take care for the patches of size 256 x 256 x 5
        """
        sample = "test"

        if self.transform:
            sample = self.transform(sample)

        return sample

train_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 SliceDataset(r"\test"), # Supply here all Volume paths
                 SliceDataset(r"\test2"),
             ),
             batch_size=batch_size,
             shuffle=True,
             num_workers=number_of_gpus,
             pin_memory=True, # loads them directly in cuda pinned memory 
             drop_last=True) # drop the last incomplete batch




for idx, input_seq in enumerate(train_loader):
    pass 



class CNN_AICT(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.startLayer = nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1, padding_mode="reflect"),
            nn.ReLU()
        )
        self.middleLayer = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.endLayer = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1, padding_mode="reflect")
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out1 = self.startLayer(x)
        out2 = self.middleLayer(out1)
        out3 = self.endLayer(out2)
        return out3

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(2), x.size(3))
        m = nn.Upsample(size=(256,256), mode='nearest')
        x = x.repeat(5, 1, 1)
        x = x.unsqueeze(0)
        x = m(x)
        print("Shape:",x.shape)
        
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer


dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=1)

# init model
cnn = CNN_AICT()
cnn.to(device)

trainer = pl.Trainer(gpus=number_of_gpus, num_nodes=number_of_nodes)
trainer.fit(cnn, train_loader)

