import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from visualization import make_grid
class CNN_AICT(pl.LightningModule):

    def __init__(self, ref_img=None):
        super().__init__()
        self.ref_img = ref_img
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

    def on_train_start(self) -> None:
        sampleImg=torch.rand((1,5,256,256)) #sample image for graph
        self.logger.experiment.add_graph(CNN_AICT(),sampleImg)
        return super().on_train_start()

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
        
    def show_activations(self, x):
        if x is not None:
            # logging reference input image       
            self.logger.experiment.add_image("input",torch.Tensor.cpu(x[0][2]),self.current_epoch,dataformats="HW")

            # logging start layer activations       
            out = self.startLayer(x)
            grid = make_grid(out,8)

            self.logger.experiment.add_image("startLayer", grid, self.current_epoch,dataformats="HW")

            # logging middle layer activations     
            out = self.middleLayer(out)
            grid = make_grid(out,8)
            self.logger.experiment.add_image("middleLayer", grid, self.current_epoch,dataformats="HW")

            # logging end layer activations    
            out = self.endLayer(out)
            grid = make_grid(out,1)
            self.logger.experiment.add_image("endLayer", grid, self.current_epoch,dataformats="HW")

    def training_epoch_end(self, outputs) -> None:
        self.show_activations(self.ref_img)
        return super().training_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer