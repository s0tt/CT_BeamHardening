import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--file-path", "-f", required=True,
                    help="Path to the folder where the .tiff files are inside")
parser.add_argument("--clim-min", "-cmin", required=False, default=0, type=int,
                    help="Path to the folder where the .tiff files are inside")
parser.add_argument("--clim-max", "-cmax", required=False, default=20000, type=int,
                    help="Path to the folder where the .tiff files are inside")

args = parser.parse_args()
img = np.array(plt.imread(args.file_path), dtype=np.float32)
plt.imshow(img, cmap="gray")
plt.clim(args.clim_min, args.clim_max)
plt.colorbar()
plt.title(str(args.file_path).split("/")[-1])
plt.savefig(args.file_path+".jpg")
plt.show()