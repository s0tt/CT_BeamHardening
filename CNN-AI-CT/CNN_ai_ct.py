import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import PSNR, MeanAbsoluteError
from torchmetrics import MetricCollection
from visualization import make_grid, plot_pred_gt, plot_ct

class CNN_AICT(pl.LightningModule):

    def __init__(self, ref_img=None):
        super().__init__()
        self.ref_img = ref_img
        self.train_metrics = MetricCollection([PSNR(), MeanAbsoluteError()])
        self.val_metrics = MetricCollection([PSNR(), MeanAbsoluteError()])
        self.test_metrics = MetricCollection([PSNR(), MeanAbsoluteError()])

        self.startLayer = nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1, padding_mode="reflect"),
            nn.ReLU()
        )

        self.middleLayer = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #5
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #6
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #9
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #13
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"), #15
            nn.BatchNorm2d(64),
            nn.ReLU(),            
        )


        self.endLayer = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1, padding_mode="reflect")
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.startLayer(x)
        out = self.middleLayer(out)
        out = self.endLayer(out)

        # calculate residual as inference output
        x_2 = torch.unsqueeze(x[:,2,:,:], dim=1) # get input middle slices
        residual = x_2 - out # from input image subtract predicted artefacts
        return residual

    def on_train_start(self) -> None:
        sampleImg=torch.rand((1,5,256,256)) #sample image for graph
        self.logger.experiment.add_graph(CNN_AICT(),sampleImg)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        residual = self(x)
   
        # get label image without neighbour slices
        y_2 = torch.unsqueeze(y[:,2,:,:], dim=1)

        loss = F.mse_loss(residual, y_2)
        return {'loss': loss, 'preds': residual, 'target': y_2}

    def train_step_end(self, outputs):
        metric_vals = self.train_metrics(outputs["preds"], outputs["target"])
        self.log_dict({
            'train_loss': outputs["loss"],
            'train_psnr': metric_vals["PSNR"],
            'train_mean_abs_err': metric_vals["MeanAbsoluteError"]
        })

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        residual = self(x)

        # get label image without neighbour slices
        y_2 = torch.unsqueeze(y[:,2,:,:], dim=1)

        loss = F.mse_loss(residual, y_2)
        return {'loss': loss, 'preds': residual, 'target': y_2}

    def validation_step_end(self, outputs):
        metric_vals = self.val_metrics(outputs["preds"], outputs["target"])
        self.log_dict({
            'val_loss': outputs["loss"],
            'val_psnr': metric_vals["PSNR"],
            'val_mean_abs_err': metric_vals["MeanAbsoluteError"]
        })

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        residual = self(x)

        # get label image without neighbour slices
        y_2 = torch.unsqueeze(y[:,2,:,:], dim=1)

        loss = F.mse_loss(residual, y_2)
        
        return {'loss': loss, 'preds': residual, 'target': y_2}

    def test_step_end(self, outputs):
        metric_vals = self.test_metrics(outputs["preds"], outputs["target"])
        self.log_dict({
            'test_loss': outputs["loss"],
            'test_psnr': metric_vals["PSNR"],
            'test_mean_abs_err': metric_vals["MeanAbsoluteError"]
        })
        
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
        self.show_activations(self.ref_img[0].type_as(outputs[0]["preds"]))
        for idx in range(self.ref_img[0].shape[0]):
            pred = self.ref_img[0][idx, :, :, :]
            gt = self.ref_img[1][idx, :, :, :]
            self.show_pred_gt(pred.type_as(outputs[0]["preds"]),
                            gt.type_as(outputs[0]["preds"]), 
                            name="ref_img_"+str(idx))
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer