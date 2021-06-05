import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import PSNR, MeanAbsoluteError
from visualization import make_grid, plot_pred_gt, plot_ct

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
        y_2 = torch.unsqueeze(y[:,2,:,:], dim=1)

        # calculate loss from ground-trouth with input image - predicted residual artifact
        residual = x_2-y_hat
        loss = F.mse_loss(residual, y_2)
        psnr_err = PSNR()
        psnr = psnr_err(residual, y_2)
        mean_abs_err = MeanAbsoluteError()
        mae = mean_abs_err(residual, y_2)
        self.log_dict({
            'train_loss': loss,
            'psnr': psnr,
            'mean_abs_err': mae
        })
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)

        # get input image without neighbour slices
        x_2 = torch.unsqueeze(x[:,2,:,:], dim=1)
        y_2 = torch.unsqueeze(y[:,2,:,:], dim=1)

        # calculate loss from ground-trouth with input image - predicted residual artifact
        residual = x_2-y_hat
        loss = F.mse_loss(residual, y_2)
        psnr_err = PSNR()
        psnr = psnr_err(residual, y_2)
        mean_abs_err = MeanAbsoluteError()
        mae = mean_abs_err(residual, y_2)

        self.log_dict({
            'val_loss': loss,
            'psnr': psnr,
            'mean_abs_err': mae
        })

        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)

        # get input image without neighbour slices
        x_2 = torch.unsqueeze(x[:,2,:,:], dim=1)
        y_2 = torch.unsqueeze(y[:,2,:,:], dim=1)

        # calculate loss from ground-trouth with input image - predicted residual artifact
        residual = x_2-y_hat
        loss = F.mse_loss(residual, y_2)
        acc = 0
        psnr_err = PSNR()
        psnr = psnr_err(residual, y_2)
        mean_abs_err = MeanAbsoluteError()
        mae = mean_abs_err(residual, y_2)

        self.log_dict({
            'test_loss': loss,
            'psnr': psnr,
            'mean_abs_err': mae
        })
        return loss
        
    def show_activations(self, x):
        if x is not None:
            # logging reference input image
            input_fig = plot_ct(x[0,2,:,:])
            self.logger.experiment.add_figure("input_img", input_fig, global_step=self.current_epoch)

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
            output_fig = plot_ct(out[0,0,:,:])
            self.logger.experiment.add_figure("endLayer", output_fig, global_step=self.current_epoch)


    def show_pred_gt(self, x, y, name="pred_gt"):
        x = torch.unsqueeze(x, dim=0)
        y = torch.unsqueeze(y, dim=0)
        x_2 = torch.unsqueeze(x[:,2,:,:], dim=1)
        y_2 = torch.unsqueeze(y[:, 2,:,:], dim=1)
        
        y_hat = self(x)
        residual = x_2-y_hat

        fig = plot_pred_gt(residual, y_2)
        self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch, close=True, walltime=None)

    def training_epoch_end(self, outputs) -> None:
        self.show_activations(self.ref_img[0])
        for idx in range(self.ref_img[0].shape[0]):
            self.show_pred_gt(self.ref_img[0][idx, :, :, :],self.ref_img[1][idx, :, :, :], name="ref_img_"+str(idx))
        return super().training_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer