
from dataloader import get_dataloader
from pytorch_lightning import loggers as pl_loggers
import torch
import pytorch_lightning as pl
from CNN_ai_ct import CNN_AICT

device = "cuda" if torch.cuda.is_available() else "cpu"
number_of_nodes = 1 
number_of_gpus = 0
num_workers = 2
batch_size = 16
dataset_stride = 128 
num_pixel = 256 
dataset_paths = [("data_path", "ground_truth_path")] 

# initialize tesnorboard logger
tb_logger = pl_loggers.TensorBoardLogger('logs/')

# get dataloader for CT input and ground-truth slices in Y-dimension
train_loader = get_dataloader(batch_size, num_workers, num_pixel, dataset_stride, dataset_paths)

# init model
cnn = CNN_AICT()
cnn.to(device)

# train module
trainer = pl.Trainer(gpus=number_of_gpus, num_nodes=number_of_nodes, logger=tb_logger)
trainer.fit(cnn, train_loader)

