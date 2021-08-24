import argparse
import os
import h5py
import numpy as np
import torch

from CNN_ai_ct import CNN_AICT
from IRR_CNN_ai_ct import IRR_CNN_AICT
from Unet import Unet
from dataloader import VolumeDatasetInfere

DATATYPE_USED = "float32"


def add_headers_and_metadata(path_old: str, path_new: str):
    """
        Adds all headers and metadata from the old volume out of the 
        volume dataset to the newly created volume. 
    """
    with h5py.File(path_old, "r") as f_in:
        with h5py.File(path_new, "r+") as f_out:
            for key in f_in.keys():
                if key != "Volume":
                    # Get parent group name for copy
                    group_path = f_in[key].parent.name
                    # Check existence of group, else create group+parent
                    group_id = f_out.require_group(group_path)
                    f_in.copy(key, group_id, group_path+key)


def runInference(model, chkpt_path, input_volume, output_path, forward_iterations, device):
    # works at the moment just with CNN-AI-CT
    if str(model).lower() == "cnn-ai-ct":
        model = CNN_AICT.load_from_checkpoint(chkpt_path)
    elif str(model).lower() == "unet":
        model = Unet.load_from_checkpoint(chkpt_path)
    elif str(model).lower() == "irr-cnn-ai-ct":
        model = IRR_CNN_AICT(
            chkpt_path, forward_iterations=forward_iterations),

    model.eval()

    model.to(device)

    # use maybe later
    inputData = VolumeDatasetInfere(
        input_volume, num_pixel=256, neighbour_img=[-2, 3], transform=None)

    with h5py.File(input_volume, "r") as f_in:
        with h5py.File(output_path, "w") as f_out:
            volume_bh = f_in['Volume']
            new_volume_dataset = None

            # Volume iteration loop (this is written for the transposed input volumes)
            for x_slice in range(2, volume_bh.shape[0]-2):
                bh_slice = volume_bh[x_slice-2: x_slice+3, :, :]

                out_slice = np.squeeze(model(torch.tensor(
                    np.expand_dims(bh_slice, axis=0))).detach().numpy(), axis=0)

                if new_volume_dataset is None:
                    #bh_slice = np.expand_dims(out_slice, axis=0)
                    new_volume_dataset = f_out.create_dataset("Volume",
                                                              data=out_slice, dtype=DATATYPE_USED, chunks=True,
                                                              maxshape=(None, None, None))
                else:
                    new_volume_dataset.resize(
                        new_volume_dataset.shape[0] + 1, axis=0)
                    new_volume_dataset[x_slice-2, :, :] = out_slice

            new_volume_dataset.attrs['MATLAB_class'] = 'double'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-point", "-cp", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", "-m", required=True, default="cnn-ai-ct",
                        help="model name [cnn-ai-ct, unet, irr-cnn-ai-ct]")
    parser.add_argument("--device", "-d", required=False, default="cpu",
                        help="Device where to run the inference. E.g. : cpu/gpu")
    parser.add_argument("--input-volume", "-i", required=True, type=str,
                        help="Path to the input volume with beam-hardening (.hdf5 file)")
    parser.add_argument("--output-path", "-o", required=True, type=str,
                        help="Path where the output volume should be exported (.hdf5 file)")
    parser.add_argument("--forward-iterations", "-fi", required=False, default=10,
                        help="Number of forward iterations: See IRR-Networks for details")
    args = parser.parse_args()

    runInference(args.model, args.check_point, args.input_volume,
                 args.output_path, args.forward_iterations, args.device)

    add_headers_and_metadata(args.input_volume, args.output_path)


if __name__ == "__main__":
    main()
