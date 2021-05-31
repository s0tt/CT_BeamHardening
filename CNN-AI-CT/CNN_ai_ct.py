import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

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
        y_hat = self(x)

        # get input image without neighbour slices
        x_2 = torch.unsqueeze(x[:,2,:,:], dim=1)

        # calculate loss from ground-trouth with input image - predicted residual artifact
        loss = F.mse_loss(y, x_2-y_hat)

        self.logger.experiment.add_scalar('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)

        # get input image without neighbour slices
        x_2 = torch.unsqueeze(x[:,2,:,:], dim=1)

        # calculate loss from ground-trouth with input image - predicted residual artifact
        loss = F.mse_loss(y, x_2-y_hat)

        self.logger.experiment.add_scalar('val_loss', loss)
        #TODO: Add validation accuracy 
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)

        # get input image without neighbour slices
        x_2 = torch.unsqueeze(x[:,2,:,:], dim=1)

        # calculate loss from ground-trouth with input image - predicted residual artifact
        loss = F.mse_loss(y, x_2-y_hat)

        self.logger.experiment.add_scalar('test_loss', loss)
        #TODO: Add test accuracy 
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer