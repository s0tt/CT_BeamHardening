
import numpy as np
import argparse
import torch
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.accelerators import accelerator

from CNN_ai_ct import CNN_AICT
from dataloader import get_dataloader
from dataloader import CtVolumeData
from ref_idx import cable_holder_noisy, cable_holder_ref
from utils import parse_dataset_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", "-f", required=True,
                        help="Path to input volume for training or JSON file with paths")
    parser.add_argument("--file-gt", "-gt", required=False,
                        help="Path to input volume ground truth file, if not provided --file-in must be a JSON")
    parser.add_argument("--nr_workers", "-w", required=False, default=2,
                        help="number of worker subproccesses to prefetch")
    parser.add_argument("--dir", "-d", required=False, default="",
                        help="directory where training artefacts are saved")
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Accelerator
    # 'ddp': multiple-gpus across many machines (python script based))
    # 'dp' : is DataParallel (split batch among GPUs of same machine)
    accelerator_type = args.accelerator
    num_workers = int(args.nr_workers) if args.nr_workers != None else None
    batch_size = 16
    dataset_stride = 128 
    num_pixel = 256 
    test_split = 0.1
    val_split = 0.2
    dataset_paths = parse_dataset_paths(args.file_in, args.file_gt)
    number_of_nodes = int(args.num_nodes) if args.num_nodes  != None else None  # number of GPU nodes for distributed training.
    number_of_gpus = int(args.gpus) if args.gpus  != None else None # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
    max_epochs = int(args.max_epochs) if args.max_epochs != None else None# max number of epochs

    # initialize tesnorboard logger
    path_log = os.path.join(args.dir, "logs")
    tb_logger = TensorBoardLogger(path_log, default_hp_metric=False)

    # init checkpoints
    val_loss_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=path_log,
        filename='CNN-AI-CT-{epoch:02d}-{val_loss:.2f}',
        mode='min',
    )

    # train_loss_callback = ModelCheckpoint(
    #     monitor='train_loss',
    #     dirpath=path_log,
    #     filename='CNN-AI-CT-{epoch:02d}-{train_loss:.2f}',
    #     mode='min',
    # )

    # train model
    trainer = pl.Trainer.from_argparse_args(
        parser, 
        logger=tb_logger,
        log_every_n_steps = 10,
        accelerator=accelerator_type,
        callbacks=[val_loss_callback],
        
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
        val_split = val_split,
        remove_noisy = np.array(cable_holder_noisy),
        manual_test = None # np.array(cable_holder_ref)
        )

    # init model
    loader = ct_volumes.val_dataloader() # ct_volumes.train_dataloader(override_batch_size=len(cable_holder_ref))
    img_test, gt = next(iter(loader)) # grab first batch for visualization

    cnn = CNN_AICT(ref_img=[img_test, gt]) # pass batch for visualization to CNN
    cnn.to(device)
    
    trainer.fit(cnn, datamodule=ct_volumes)

    # test model
    trainer.test(datamodule=ct_volumes, ckpt_path=val_loss_callback.best_model_path)
    # trainer.test(datamodule=ct_volumes, ckpt_path=train_loss_callback.best_model_path)

if __name__ == "__main__":
    main()