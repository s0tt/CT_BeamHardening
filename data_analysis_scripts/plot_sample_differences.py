import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../CNN-AI-CT'))

from dataloader import VolumeDataset
from utils import parse_dataset_paths

def plot_slice(x, gt, dataset_name, out_path, index):
    # x: Poly, gt: Mono

    fig, axs = plt.subplots(2, 2, figsize=(14, 15))

    max_first_row = max([x.max(), gt.max()])
    min_first_row = min([x.min(), gt.min()])
    fig.subplots_adjust(left=0.05, bottom=0.06, right=0.95,
                        top=0.94, wspace=0.18, hspace=.1)

    im = axs[0, 0].imshow(x, vmin=min_first_row,
                          vmax=max_first_row, cmap="gray")
    axs[0, 0].set_title("Input (Poly) | Mean: %.2f" % x.mean(), size=20)

    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=13)

    im = axs[0, 1].imshow(gt, vmin=min_first_row,
                          vmax=max_first_row, cmap="gray")
    axs[0, 1].set_title("Gt (Mono) | Mean: %.2f" % gt.mean(), size=20)

    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=13)

    im = axs[1, 0].imshow(x-gt, cmap="gray")
    axs[1, 0].set_title("Input (Poly) - GT (Mono) | Mean: %.2f" %
                        (x-gt).mean(), size=20)

    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=13)

    axs[1, 1].hist((x-gt).flatten(), 100,
                   density=True, rwidth=0.8, alpha=0.7)
    axs[1, 1].set_title("Hist: Input (Poly) - GT (Mono)", size=17)
    axs[1, 1].grid(True)

    plt.savefig(os.path.join(out_path, dataset_name,
                'sample_nr_' + str(index) + '.png'))

    plt.close('all')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", "-f", required=True,
                        help="Path to json file that contains all datasets")
    parser.add_argument("--dataset-names", "-dn", required=False, nargs='+', default=["all"],
                        help="Names of the datasets of --file-in that should analysed")
    parser.add_argument("--output-folder", "-o", required=True,
                        help="Folder path in which all analysis data should be written")
    parser.add_argument("--dataset-stride", "-s", required=False, default=128, type=int,
                        help="Dataset stride")
    parser.add_argument("--num_pixels", "-np", required=False, default=256, type=int,
                        help="Width of slice")
    parser.add_argument("--step-size", "-ss", required=False, default=1000, type=int,
                        help="Each |step-size| sample is written out")
    args = parser.parse_args()

    dataset_paths = parse_dataset_paths(args.file_in, args.dataset_names)


    for paths in dataset_paths:
        dataset = VolumeDataset(
            paths[0], paths[1], args.num_pixels, args.dataset_stride)
        dataset_length = len(dataset)

        os.makedirs(os.path.join(args.output_folder, paths[2]), exist_ok=True)

        index = 0
        while(index < dataset_length):
            sample = dataset.__getitem__(index)

            poly_slice = sample[0][2, :, :]
            mono_slice = sample[1][2, :, :]

            plot_slice(poly_slice, mono_slice,
                       paths[2], args.output_folder, index)

            index += args.step_size


if __name__ == "__main__":
    main()
