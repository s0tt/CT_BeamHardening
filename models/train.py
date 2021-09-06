
import numpy as np
import argparse
import torch
import os
import json
import datetime
import subprocess
import itertools

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.accelerators import accelerator
from torch.nn.modules.activation import Threshold
from pytorch_lightning.plugins import DDPPlugin
import torchmetrics


from CNN_ai_ct import CNN_AICT
from CNN_ai_ct_skip import CNN_AICT_SKIP
from CNN_ai_ct_silu import CNN_AICT_SILU
from IRR_CNN_ai_ct import IRR_CNN_AICT
from Unet import Unet
from dataloader import CtVolumeData, update_noisy_indexes, get_noisy_indexes
from utils import parse_dataset_paths, add_datasets_to_noisy_images_json


def get_git_revision_short_hash(path):
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=path).decode('ascii').strip()
        git_branch = subprocess.check_output(
            ['git', 'branch', '-vv'], cwd=path).decode('ascii').strip()
        for str_branch in git_branch.split('\n'):
            if str_branch.find('*') != -1:
                git_branch = str_branch
                break
    except:
        git_hash = "UNKNOWN"
        git_branch = "UNKNOWN"
    return git_hash, git_branch


def switch_model(model_str):
    neighbour_img = None
    if str(model_str).lower() == "cnn-ai-ct":
        # defines range of neighbour slices e.g. here -2 to +3 --> 5 slices
        neighbour_img = [-2, 3]
    elif str(model_str).lower() == "unet":
        # defines range of neighbour slices e.g. here 0 to 1 --> 1 slice
        neighbour_img = [0, 1]

    return neighbour_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", "-f", required=True,
                        help="Path to json file that contains all datasets")
    parser.add_argument("--model", "-m", required=True, default="cnn-ai-ct",
                        help="model name [cnn-ai-ct, unet, irr-cnn-ai-ct, cnn-ai-ct-silu, cnn-ai-ct-skip]")
    parser.add_argument("--batch-size", "-bs", required=True, default=16,
                        help="Batch size")
    parser.add_argument("--dataset-names", "-dn", required=False, nargs='+', default=["all"],
                        help="Names of the datasets of --file-in that should be used for training")
    parser.add_argument("--file-noisy-indexes", "-nf", required=False,
                        help="Path to the json file that contains the noisy indexes")
    parser.add_argument("--nr_workers", "-w", required=False, default=2,
                        help="number of worker subproccesses to prefetch")
    parser.add_argument("--dir", "-d", required=False, default="",
                        help="directory where training artefacts are saved")
    parser.add_argument("--forward-iterations", "-fi", required=False, default=10,
                        help="Number of forward iterations: See IRR-Networks for details")
    parser.add_argument("--remove-noisy-slices", "-rn", required=False, default=None,
                        help="Parameter to activate/ deactive the removement of noisy slices")
    parser.add_argument("--plot-test-nr", "-pt", required=False, default=10,
                        help="number of images to plot from test set")
    parser.add_argument("--plot-weights", "-pw", required=False, action="store_true", default=False,
                        help="If argument is given (-pw) plot model weights")
    parser.add_argument("--plot-val-nr", "-pv", required=False, default=None,
                        help="number of images to plot from val set in each validation epoch")
    parser.add_argument("--tb-name", "-tn", required=False, default="default",
                        help="name of tensorboard experiment")
    parser.add_argument("--custom-init", "-ci", required=False, action="store_true", default=False,
                        help="If argument is given (-ci) custom init the model weights")
    parser.add_argument("--transfer-learn-path", "-tlp", required=False, default=None,
                        help="Use transfer learning from given model checkpoint by freezing layers and retrain endLayer")
    parser.add_argument("--device", "-dv", required=False, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # initialize tensorboard logger
    path_log = os.path.join(args.dir, "logs")
    os.makedirs(os.path.join(path_log, args.tb_name), exist_ok=True)

    tb_logger = TensorBoardLogger(
        path_log, name=args.tb_name, default_hp_metric=False)
    # os.makedirs(tb_logger.log_dir, exist_ok=True)

    # Accelerator
    # 'ddp': multiple-gpus across many machines (python script based))
    # 'dp' : is DataParallel (split batch among GPUs of same machine)
    num_workers = int(args.nr_workers) if args.nr_workers != None else None
    batch_size = int(args.batch_size) if args.batch_size != None else None
    dataset_stride = 128
    num_pixel = 256
    test_split = 0.1
    val_split = 0.2
    noise_removal_threshold = 2.5
    dataset_paths = parse_dataset_paths(args.file_in, args.dataset_names)
    if args.remove_noisy_slices:
        print("Calculate and remove noisy indices")
        add_datasets_to_noisy_images_json(
            args.file_in, args.file_noisy_indexes)
        update_noisy_indexes(num_pixel, dataset_stride, dataset_paths,
                             args.file_noisy_indexes, threshold=noise_removal_threshold)

    # number of GPU nodes for distributed training.
    number_of_nodes = int(args.num_nodes) if args.num_nodes != None else None
    # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
    number_of_gpus = int(args.gpus) if args.gpus != None else None
    # max number of epochs
    max_epochs = int(args.max_epochs) if args.max_epochs != None else None

    # get model specific parts
    neighbour_img = switch_model(str(args.model))

    # init checkpoints
    val_loss_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=tb_logger.log_dir,
        filename=str(args.model)+'-{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        save_top_k=2,
        mode='min',
    )

    train_loss_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=tb_logger.log_dir,
        filename=str(args.model)+'-{epoch:02d}-{train_loss:.2f}',
        save_last=True,
        save_top_k=2,
        mode='min',
    )

    if args.remove_noisy_slices:
        noisy_indexes = get_noisy_indexes(args.file_noisy_indexes)
    else:
        noisy_indexes = None

    # CT data loading
    ct_volumes = CtVolumeData(
        paths=dataset_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_stride=dataset_stride,
        num_pixel=num_pixel,
        test_split=test_split,
        val_split=val_split,
        noisy_indexes=None,
        manual_test=None,
        neighbour_img=neighbour_img
    )

    # init model
    # ct_volumes.train_dataloader(override_batch_size=len(cable_holder_ref))
    loader = ct_volumes.val_dataloader()
    img_test, gt = next(iter(loader))  # grab first batch for visualization

    if str(args.model).lower() == "cnn-ai-ct":
        if args.transfer_learn_path is None:
            model = CNN_AICT(ref_img=[img_test, gt], plot_test_step=args.plot_test_nr,
                             plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights, custom_init=args.custom_init)
        else:
            model = CNN_AICT.load_from_checkpoint(args.transfer_learn_path)
            # freeze start and middle layers for transfer-learning/fine-tuning of the endLayer to new data
            for param in itertools.chain(model.startLayer.parameters(), model.middleLayer.parameters()):
                param.requires_grad = False

        plugin = DDPPlugin(find_unused_parameters=False)

    elif str(args.model).lower() == "unet":
        model = Unet(ref_img=[img_test, gt], plot_test_step=args.plot_test_nr,
                     plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights)
        plugin = DDPPlugin(find_unused_parameters=True)
    elif str(args.model).lower() == "irr-cnn-ai-ct":
        model = IRR_CNN_AICT(forward_iterations=int(args.forward_iterations), ref_img=[img_test, gt], plot_test_step=args.plot_test_nr,
                             plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights)  # pass batch for visualization to CNN
        plugin = DDPPlugin(find_unused_parameters=True)
    elif str(args.model).lower() == "cnn-ai-ct-silu":
        model = CNN_AICT_SILU(ref_img=[img_test, gt], plot_test_step=args.plot_test_nr,
                              plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights)
        plugin = DDPPlugin(find_unused_parameters=False)

    elif str(args.model).lower() == "cnn-ai-ct-skip":
        model = CNN_AICT_SKIP(ref_img=[img_test, gt], plot_test_step=args.plot_test_nr,
                              plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights)
        plugin = DDPPlugin(find_unused_parameters=False)

    model.to(args.device)

    # construct JSON log only once for all DDP processes
    if model.global_rank == 0:
        time_str = datetime.datetime.now().strftime("%m_%d_%y__%H_%M_%S")
        os.makedirs(os.path.join(path_log, "json"), exist_ok=True)
        json_path = os.path.join(
            path_log, "json", time_str+f"_train_args.json")
        with open(json_path, "w+") as f:
            repo_path = os.path.split(args.file_in)[0]
            hash_id, branch = get_git_revision_short_hash(repo_path)
            trainDict = {}

            trainDict["Date"] = datetime.datetime.now().strftime(
                "%m-%d-%y %H:%M:%S")
            trainDict["Model"] = str(args.model)
            trainDict["Git ID"] = str(hash_id)
            trainDict["Git Branch"] = str(branch)
            trainDict["Repo Dir"] = str(repo_path)
            trainDict["Dataset Seed"] = str(ct_volumes.dataset_seed)
            trainDict["TB Log Dir"] = str(tb_logger.log_dir)
            trainDict["Data Paths"] = str([dataset[2]
                                           for dataset in dataset_paths])
            trainDict["Train/Val/Test Len."] = str([len(ct_volumes.dataset_train), len(
                ct_volumes.dataset_val), len(ct_volumes.dataset_test)])
            trainDict["Pytorch Lightning Ver"] = str(pl.__version__)
            trainDict["Torchmetrics Ver"] = str(torchmetrics.__version__)
            trainDict["Len Noisy indexes"] = str(
                len(noisy_indexes) if noisy_indexes is not None else 0)
            trainDict["Args"] = args.__dict__
            json.dump(trainDict, f, indent=4)
            f.close()

    # train model
    trainer = pl.Trainer.from_argparse_args(
        parser,
        logger=tb_logger,
        log_every_n_steps=10,
        callbacks=[train_loss_callback, val_loss_callback],
        plugins=plugin
    )

    # fit model
    trainer.fit(model, datamodule=ct_volumes)

    # test model
    trainer.test(datamodule=ct_volumes,
                 ckpt_path=val_loss_callback.best_model_path)
    # trainer.test(datamodule=ct_volumes, ckpt_path=train_loss_callback.best_model_path)


if __name__ == "__main__":
    main()
