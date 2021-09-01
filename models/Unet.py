import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.modules import padding
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics import PSNR
from torchmetrics import MetricCollection
from visualization import make_grid, plot_pred_gt, plot_ct


class Unet(pl.LightningModule):

    def __init__(self, ref_img=None, plot_test_step=None, plot_val_step=None, plot_weights=False):
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

        def double_conv(input, out):
            return nn.Sequential(
                nn.Conv2d(input, out, kernel_size=3, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
                nn.Conv2d(out, out, kernel_size=3, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
            )

        class Encoder(nn.Module):
            def __init__(self, input_conv, input_max, out_conv, out):
                super().__init__()
                self.conv_in = nn.Conv2d(input_conv, out_conv, 3, padding=1)
                self.conv_d = double_conv(input_max+out_conv, out)
                self.avg_pool = nn.AvgPool2d(2, stride=2)
                self.max_pool = nn.MaxPool2d(2, stride=2)

            def forward(self, input_avg, input_max):
                avg = self.avg_pool(input_avg)
                max = self.max_pool(input_max)
                x = self.conv_in(avg)
                x = torch.cat([x, max], dim=1)
                x = self.conv_d(x)
                return avg, x

        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        class up(nn.Module):
            def __init__(self, input_up, input_cat, out, bilinear=False):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranspose2d(input_up, input_up,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(input_up+input_cat, out)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1)  # why 1?
                return self.conv(x)

        self.in_conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
        )
        self.out_conv = double_conv(20, 1)

        self.encoder1 = Encoder(1, 4, 8, 40)
        self.encoder2 = Encoder(1, 40, 16, 80)
        self.encoder3 = Encoder(1, 80, 32, 160)
        self.encoder4 = Encoder(1, 160, 64, 320)
        self.encoder5 = Encoder(1, 320, 96, 480)
        self.encoder6 = Encoder(1, 480, 192, 960)

        self.up1 = up(960, 480, 480)
        self.up2 = up(480, 320, 320)
        self.up3 = up(320, 160, 160)
        self.up4 = up(160, 80, 80)
        self.up5 = up(80, 40, 40)
        self.up6 = up(40, 4, 20)

    def forward(self, x):
        # encode
        avg = x
        x0 = self.in_conv(x)
        avg, x1 = self.encoder1(avg, x0)
        avg, x2 = self.encoder2(avg, x1)
        avg, x3 = self.encoder3(avg, x2)
        avg, x4 = self.encoder4(avg, x3)
        avg, x5 = self.encoder5(avg, x4)
        avg, x6 = self.encoder6(avg, x5)

        # decode
        x = self.up1(x6,x5)
        x = self.up2(x5,x4)
        x = self.up3(x,x3)
        x = self.up4(x,x2)
        x = self.up5(x,x1)
        x = self.up6(x,x0)
        x = self.out_conv(x)
        return x

    def on_train_start(self) -> None:
        sampleImg = torch.rand((1, 1, 256, 256))  # sample image for graph
        self.logger.experiment.add_graph(Unet(), sampleImg)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)

        self.log_dict(self.train_metrics(y_hat, y))
        self.log('train_loss', loss, sync_dist=True)
        self.logger.experiment.add_scalars(
            "losses", {"train_loss": loss}, global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)

        self.log_dict(self.val_metrics(y_hat, y))
        self.log('val_loss', loss, sync_dist=True)
        self.logger.experiment.add_scalars(
            "losses", {"val_loss": loss}, global_step=self.global_step)

        if self.plot_val_step is not None:
            # plot n-val images
            for idx in range(x.shape[0]):
                if self.plot_val_cnt > self.plot_val_step:
                    break
                self.show_pred_gt(x[idx, :, :, :],
                                  y[idx, :, :, :],
                                  y_hat_in=y_hat[idx, :, :, :],
                                  name=("val_img_"+str(self.plot_val_cnt)),
                                  use_global_step=True)
                self.plot_val_cnt += 1

    def on_validation_end(self) -> None:
        self.plot_val_cnt = 0

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)

        self.log_dict(self.test_metrics(y_hat, y))
        self.log('test_loss', loss, sync_dist=True)

        if self.plot_test_step is not None:
            # plot n-test images
            for idx in range(x.shape[0]):
                if self.plot_test_cnt > self.plot_test_step:
                    break
                self.show_pred_gt(x[idx, :, :, :],
                                  y[idx, :, :, :],
                                  y_hat_in=y_hat[idx, :, :, :],
                                  name=("test_img_"+str(self.plot_test_cnt)))
                self.plot_test_cnt += 1

    def show_pred_gt(self, x, y, y_hat_in=None, name="pred_gt", use_global_step=False):
        x = torch.unsqueeze(x, dim=0)
        y = torch.unsqueeze(y, dim=0)
        x_2 = torch.unsqueeze(x[:, :, :, :], dim=1)
        y_2 = torch.unsqueeze(y[:, :, :, :], dim=1)

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
        # TODO: Adjust activation plot to Unet
        #self.show_activations(self.ref_img[0].type_as(outputs[0]["loss"]))

        # for all reference images plot model prediction after epoch
        for idx in range(self.ref_img[0].shape[0]):
            pred = self.ref_img[0][idx, :, :, :]
            gt = self.ref_img[1][idx, :, :, :]
            self.show_pred_gt(pred.type_as(outputs[0]["loss"]),
                              gt.type_as(outputs[0]["loss"]),
                              name="ref_img_"+str(idx))

        # plot model filter weights after epoch
        # TODO: Adjust weight plot to Unet
        # if self.plot_weights:
        #     self.show_weights()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer
