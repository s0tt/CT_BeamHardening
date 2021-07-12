import argparse
from genericpath import exists
import os
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from visualization import make_grid, plot_pred_gt, plot_ct
from CNN_ai_ct import CNN_AICT
from dataloader import CtVolumeData
from utils import parse_dataset_paths


def runModel(chkpt_path, log_path, datasets, nr_test_samples, tensorboard_name, workers):
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

    model = CNN_AICT.load_from_checkpoint(chkpt_path)
    model.plot_test_step = nr_test_samples
    model.eval()
    experiment_name = tensorboard_name+"-"+chkpt_path.split("/")[-1]
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
    args = parser.parse_args()
    data_paths = parse_dataset_paths(args.data_path, args.dataset_names)
    runModel(args.check_point, args.log_dir, data_paths, args.test_samples, args.tensorboard_name, args.workers)


if __name__ == "__main__":
    main()
