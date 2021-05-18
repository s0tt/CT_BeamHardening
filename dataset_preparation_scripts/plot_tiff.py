import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--file-path", "-f", required=True,
                    help="Path to the folder where the .tiff files are inside")

args = parser.parse_args()
img = plt.imread(args.file_path)
plt.imshow(img, cmap="gray")
plt.colorbar()
plt.show()
