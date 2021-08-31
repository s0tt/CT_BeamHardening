import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import PSNR
from visualization import make_grid, plot_pred_gt, plot_ct


class CNN_AICT(pl.LightningModule):

    def __init__(self, ref_img=None, plot_test_step=None, plot_val_step=None, plot_weights=False, custom_init=False):
        super().__init__()
        self.ref_img = ref_img
        self.plot_test_step = plot_test_step  # n-test images shall be plotted
        self.plot_val_step = plot_val_step  # n-val images shall be plotted
        self.plot_test_cnt = 0
        self.plot_val_cnt = 0
        self.plot_weights = plot_weights
        metrics = MetricCollection([PSNR(), MeanAbsoluteError()])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.startLayer = nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1, padding_mode="reflect"),
            nn.ReLU()
        )

        self.middleLayer = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 5
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 6
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 9
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 13
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect"),  # 15
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.endLayer = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1, padding_mode="reflect")
        )
        
        if custom_init:
            self.startLayer.apply(self.weights_init)
            self.middleLayer.apply(self.weights_init)
            self.endLayer.apply(self.weights_init)


    def weights_init(self, seq):
        if type(seq) in [nn.Conv2d, nn.Linear]:
            torch.nn.init.normal_(seq.weight, mean=0.0, std=1.0)
            torch.nn.init.normal_(seq.bias, mean=0.0, std=1.0)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.startLayer(x)
        out = self.middleLayer(out)
        out = self.endLayer(out)

        # calculate residual as inference output
        x_2 = torch.unsqueeze(x[:, 2, :, :], dim=1)  # get input middle slices
        residual = x_2 - out  # from input image subtract predicted artefacts
        return residual

    def on_train_start(self) -> None:
        sampleImg = torch.rand((1, 5, 256, 256))  # sample image for graph
        self.logger.experiment.add_graph(CNN_AICT(), sampleImg)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        residual = self(x)

        # get label image without neighbour slices
        y_2 = torch.unsqueeze(y[:, 2, :, :], dim=1)

        loss = F.mse_loss(residual, y_2)

        self.log_dict(self.train_metrics(residual, y_2))
        self.log('train_loss', loss, sync_dist=True)
        self.logger.experiment.add_scalars(
            "losses", {"train_loss": loss}, global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        residual = self(x)

        # get label image without neighbour slices
        y_2 = torch.unsqueeze(y[:, 2, :, :], dim=1)

        loss = F.mse_loss(residual, y_2)

        self.log_dict(self.val_metrics(residual, y_2))
        self.log('val_loss', loss, sync_dist=True)
        self.logger.experiment.add_scalars(
            "losses", {"val_loss": loss}, global_step=self.global_step)

        if self.plot_val_step is not None:
            self.show_activations(x.type_as(loss))
            # plot n-val images
            for idx in range(x.shape[0]):
                if self.plot_val_cnt > self.plot_val_step:
                    break
                self.show_pred_gt(x[idx, :, :, :],
                                  y[idx, :, :, :],
                                  y_hat_in=residual[idx, :, :, :],
                                  name=("val_img_"+str(self.plot_val_cnt)),
                                  use_global_step=True)
                self.plot_val_cnt += 1

    def on_validation_end(self) -> None:
        self.plot_val_cnt = 0

    def test_step(self, batch, batch_idx):
        x, y = batch
        residual = self(x)

        # get label image without neighbour slices
        y_2 = torch.unsqueeze(y[:, 2, :, :], dim=1)

        loss = F.mse_loss(residual, y_2)

        self.log_dict(self.test_metrics(residual, y_2))
        self.log('test_loss', loss, sync_dist=True)

        if self.plot_test_step is not None:
            self.show_activations(x.type_as(loss))
            # plot n-test images
            for idx in range(x.shape[0]):
                if self.plot_test_cnt > self.plot_test_step:
                    break
                self.show_pred_gt(x[idx, :, :, :],
                                  y[idx, :, :, :],
                                  y_hat_in=residual[idx, :, :, :],
                                  name=("test_img_"+str(self.plot_test_cnt)))
                self.plot_test_cnt += 1

    def show_weights(self, channel_nr=[5, 64, 64]):
        # log start filter weights
        for i in range(channel_nr[0]):
            weights = self.startLayer[0].weight[:, i, :, :]
            grid = make_grid(weights, 8)
            self.logger.experiment.add_image(
                "input_weights"+str(i), grid, global_step=self.current_epoch, dataformats="HW")

        # get last conv2d from middle layers
        idx_list = []
        for idx, layer in enumerate(self.middleLayer):
            if isinstance(layer, nn.Conv2d):
                idx_list.append(idx)

        # log weights of last middle layer filters
        for i in range(channel_nr[1]):
            weights = self.middleLayer[idx_list[-1]].weight[:, i, :, :]
            grid = make_grid(weights, 8)
            self.logger.experiment.add_image(
                "middle_weights"+str(i), grid, global_step=self.current_epoch, dataformats="HW")

        # log weights of end layer
        for i in range(channel_nr[1]):
            weights = self.endLayer[0].weight[:, i, :, :]
            grid = make_grid(weights, 8)
            self.logger.experiment.add_image(
                "end_weights"+str(i), grid, global_step=self.current_epoch, dataformats="HW")

    def show_activations(self, x):
        if x is not None:
            # logging reference input image
            input_fig = plot_ct(x[0, 2, :, :])
            self.logger.experiment.add_figure(
                "input_img", input_fig, global_step=self.current_epoch)

            # logging start layer activations
            out = self.startLayer(x)
            grid = make_grid(out[0, :, :, :], 8)

            self.logger.experiment.add_image(
                "startLayer", grid, self.current_epoch, dataformats="HW")

            # logging middle layer activations
            out = self.middleLayer(out)
            grid = make_grid(out[0, :, :, :], 8)
            self.logger.experiment.add_image(
                "middleLayer", grid, self.current_epoch, dataformats="HW")

            # logging end layer activations
            out = self.endLayer(out)
            output_fig = plot_ct(out[0, 0, :, :])
            self.logger.experiment.add_figure(
                "endLayer", output_fig, global_step=self.current_epoch)

    def show_pred_gt(self, x, y, y_hat_in=None, name="pred_gt", use_global_step=False):
        x = torch.unsqueeze(x, dim=0)
        y = torch.unsqueeze(y, dim=0)
        x_2 = torch.unsqueeze(x[:, 2, :, :], dim=1)
        y_2 = torch.unsqueeze(y[:, 2, :, :], dim=1)

        if y_hat_in is None:
            y_hat = self(x)
        else:
            y_hat = y_hat_in

        fig = plot_pred_gt(x_2, y_hat, y_2)
        if use_global_step:
            self.logger.experiment.add_figure(
                name, fig, global_step=self.global_step, close=True, walltime=None)
        else:
            self.logger.experiment.add_figure(
                name, fig, global_step=self.current_epoch, close=True, walltime=None)

    def training_epoch_end(self, outputs) -> None:
        if self.ref_img is not None:
            self.show_activations(self.ref_img[0].type_as(outputs[0]["loss"]))

            # for all reference images plot model prediction after epoch
            for idx in range(self.ref_img[0].shape[0]):
                pred = self.ref_img[0][idx, :, :, :]
                gt = self.ref_img[1][idx, :, :, :]
                self.show_pred_gt(pred.type_as(outputs[0]["loss"]),
                                gt.type_as(outputs[0]["loss"]),
                                name="ref_img_"+str(idx))

        # plot model filter weights after epoch
        if self.plot_weights:
            self.show_weights()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer
