
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", "-f", required=True,
                        help="Path to input volume for training")
    parser.add_argument("--file-gt", "-gt", required=True,
                        help="Path to input volume ground truth file")
    parser.add_argument("--nr_gpus", "-g", required=False, default=0,
                        help="number of gpus to use")
    parser.add_argument("--nr_workers", "-w", required=False, default=2,
                        help="number of worker subproccesses to prefetch")
    parser.add_argument("--nr_nodes", "-n", required=False, default=1,
                        help="Path to input volume ground truth file")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    number_of_nodes = args.nr_nodes # number of GPU nodes for distributed training.
    number_of_gpus = args.nr_gpus # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
    accelerator_type = None # TODO: ('dp') is DataParallel (split batch among GPUs of same machine)
    num_workers = args.nr_workers
    batch_size = 16
    dataset_stride = 128 
    num_pixel = 256 
    test_split = 0.3
    dataset_paths = [(args.file_in, args.file_gt)] 

    # initialize tesnorboard logger
    tb_logger = TensorBoardLogger('logs/')

    # init checkpoints
    # val_loss_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath='/net/pasnas01/pool1/enpro-2021-voxie/training/cnn_ai_ct',
    #     filename='CNN-AI-CT-{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=3,
    #     mode='min',
    # )

    train_loss_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='/net/pasnas01/pool1/enpro-2021-voxie/training/cnn_ai_ct',
        filename='CNN-AI-CT-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Creating data indices for training and test splits
    loader = get_dataloader(batch_size, num_workers, num_pixel, dataset_stride, 
                                dataset_paths, shuffle=False)
    dataset_size = loader.dataset.len_datasets[0]
    indices = list(range(dataset_size))
    split = int(np.round(test_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # get dataloader for CT input and ground-truth slices in Y-dimension
    train_loader = get_dataloader(batch_size, num_workers, num_pixel, dataset_stride, dataset_paths, 
                                    sampler=train_sampler, shuffle=False)


    # init model
    cnn = CNN_AICT()
    cnn.to(device)

    # train model
    trainer = pl.Trainer(
        gpus=number_of_gpus, 
        num_nodes=number_of_nodes, 
        logger=tb_logger, 
        accelerator=accelerator_type,
        callbacks=[train_loss_callback] #[val_loss_callback, train_loss_callback]
        )
    trainer.fit(cnn, train_loader)

    # test model
    test_loader = get_dataloader(batch_size, num_workers, num_pixel, dataset_stride, 
                                    dataset_paths, sampler=test_sampler, shuffle=False)
    trainer.test(test_dataloaders=test_loader)

if __name__ == "__main__":
    main()