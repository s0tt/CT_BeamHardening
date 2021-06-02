
import numpy as np
import argparse
import torch
from torch.utils.data.sampler import SubsetRandomSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.accelerators import accelerator

from CNN_ai_ct import CNN_AICT
from dataloader import get_dataloader
from dataloader import CtVolumeData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", "-f", required=True,
                        help="Path to input volume for training")
    parser.add_argument("--file-gt", "-gt", required=True,
                        help="Path to input volume ground truth file")
    parser.add_argument("--nr_workers", "-w", required=False, default=2,
                        help="number of worker subproccesses to prefetch")
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Accelerator
    # 'ddp': multiple-gpus across many machines (python script based))
    # 'dp' : is DataParallel (split batch among GPUs of same machine)
    accelerator_type = args.accelerator
    num_workers = args.nr_workers
    batch_size = 16
    dataset_stride = 128 
    num_pixel = 256 
    test_split = 0.3
    val_split = 0.2
    dataset_paths = [(args.file_in, args.file_gt)] 
    number_of_nodes = args.num_nodes # number of GPU nodes for distributed training.
    number_of_gpus = args.gpus # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
    max_epochs = args.max_epochs # max number of epochs

    # initialize tesnorboard logger
    tb_logger = TensorBoardLogger('logs/')

    # init checkpoints
    val_loss_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/net/pasnas01/pool1/enpro-2021-voxie/training/cnn_ai_ct',
        filename='CNN-AI-CT-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    train_loss_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='/net/pasnas01/pool1/enpro-2021-voxie/training/cnn_ai_ct',
        filename='CNN-AI-CT-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # train model
    trainer = pl.Trainer.from_argparse_args(
        parser, 
        logger=tb_logger, 
        accelerator=accelerator_type,
        callbacks=[train_loss_callback, val_loss_callback]
        )

    # TODO: Add Command Line Interface (CLI)
    #cli = LightningCLI(LitClassifier, MNISTDataModule, seed_everything_default=1234)
    #result = cli.trainer.test(cli.model, datamodule=cli.datamodule)

    # CT data loading
    ct_volumes = CtVolumeData(
        paths=dataset_paths, 
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_stride = dataset_stride, 
        num_pixel = num_pixel,
        test_split = test_split,
        val_split = val_split
        )

    # init model
    ref_img, ref_label = next(iter(ct_volumes.test_dataloader()))
    cnn = CNN_AICT(ref_img=ref_img)
    cnn.to(device)
    
    trainer.fit(cnn, datamodule=ct_volumes)

    # test model
    trainer.test(datamodule=CtVolumeData)

if __name__ == "__main__":
    main()