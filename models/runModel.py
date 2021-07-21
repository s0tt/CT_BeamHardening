import argparse
from genericpath import exists
import os
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from visualization import make_grid, plot_pred_gt, plot_ct
from CNN_ai_ct import CNN_AICT
from IRR_CNN_ai_ct import IRR_CNN_AICT
from Unet import Unet
from dataloader import CtVolumeData
from utils import parse_dataset_paths


def runModel(model, chkpt_path, log_path, datasets, nr_test_samples, tensorboard_name, workers, forward_iterations):
    ct_volumes = CtVolumeData(
        paths=datasets,
        batch_size=5,
        num_workers=workers,
        dataset_stride=128,
        num_pixel=256,
        test_split=0.1,
        val_split=0.2,
        noisy_indexes=None,
        manual_test=nr_test_samples
    )

    if str(model).lower() == "cnn-ai-ct":
        model = CNN_AICT.load_from_checkpoint(chkpt_path)
    elif str(model).lower() == "unet":
        model = Unet.load_from_checkpoint(chkpt_path)
    elif str(model).lower() == "irr-cnn-ai-ct":
        model = IRR_CNN_AICT(chkpt_path, forward_iterations=forward_iterations),

    model.plot_test_step = nr_test_samples
    model.eval()
    experiment_name = tensorboard_name+"-"+chkpt_path.split("/")[-2]+"-"+chkpt_path.split("/")[-1]
    os.makedirs(os.path.join(log_path,experiment_name), exist_ok=True)
    tb_logger = TensorBoardLogger(log_path, default_hp_metric=False, name=experiment_name)
    trainer = pl.Trainer(
        logger=tb_logger,
        log_every_n_steps=1,
        plugins=DDPPlugin(find_unused_parameters=False)
    )

    trainer.test(model, datamodule=ct_volumes)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-point", "-cp", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", "-m", required=True, default="cnn-ai-ct",
                        help="model name [cnn-ai-ct, unet, irr-cnn-ai-ct]")
    parser.add_argument("--data-path", "-dp", required=True, type=str,
                        help="Path to dataset json")
    parser.add_argument("--log-dir", "-ld", required=False, default="",
                        help="directory where training artefacts are saved")
    parser.add_argument("--dataset-names", "-dn", required=True, type=str,
                        help="name of datasets to load")
    parser.add_argument("--test-samples", "-ts", required=False, default=25, type=int,
                        help="name of datasets to load")
    parser.add_argument("--tensorboard-name", "-tn", required=False, default="default", type=str,
                        help="name of datasets to load")
    parser.add_argument("--workers", "-w", required=False, default=8, type=int,
                        help="name of datasets to load")
    parser.add_argument("--forward-iterations", "-fi", required=False, default=10,
                        help="Number of forward iterations: See IRR-Networks for details")
    args = parser.parse_args()
    data_paths = parse_dataset_paths(args.data_path, args.dataset_names)
    runModel(args.model, args.check_point, args.log_dir, data_paths, args.test_samples, args.tensorboard_name, args.workers, 
            args.forward_iterations)


if __name__ == "__main__":
    main()
