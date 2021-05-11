import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
#from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import numpy as np
from PIL import Image 

device = "cuda" if torch.cuda.is_available() else "cpu"
number_of_nodes = 1 
number_of_gpus = 0 

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
        )
        self.endLayer = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1, padding_mode="reflect")
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.startLayer(x)
        for i in range(15):
            out = self.middleLayer(out)
        out3 = self.endLayer(out)
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

