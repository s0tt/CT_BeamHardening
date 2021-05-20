import numpy as np
import argparse
import tifffile
import matplotlib.pyplot as plt
import os

def ffc(img_ct, img_white):
    img_shape = np.array(img_ct).shape

    # take zeros as reference black image
    black = np.zeros((img_shape), dtype=np.float64)

    # take reference white image from aRTist
    white = np.array(img_white, dtype=np.float64)

    # flat-field-correction (ffc)
    img_new = ((img_ct - black) / (white-black))

    # cut small values
    img_new = np.maximum((1/60000), img_new)

    return img_new


def conv_attenuation(img):
    # take mean of upper left image square as zero intensity
    i_0 = np.mean(img[0:250, 0:250])
    img_att = -np.log(img/i_0)
    return img_att


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", "-f", required=True,
                        help="Path to the folder where the .tiff files are inside")

    parser.add_argument("--output-path", "-o", required=True, type=str,
                        help="Path were processed images shall be saved to")

    parser.add_argument("--white-path", "-w", required=True, type=str,
                        help="path of white detector reference .tiff file")

    parser.add_argument("--overwrite", "-owr", required=False, type=int, default=0,
                        help="overwrite original files with new 0=True, 1=False \
                        (default:0)")
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.file_path):
        img_white = np.array(tifffile.imread(args.white_path), dtype=np.float64)
        for name in sorted(files):
            img = np.array(tifffile.imread(os.path.join(root, name)), dtype=np.float64)
            img = ffc(img, img_white)
            img = conv_attenuation(img)

            if args.overwrite == 1:
                #img_pil.save(os.path.join(root, name))
                tifffile.imsave(os.path.join(root, name), img, dtype=np.float64)
            else:
                #img_pil.save(os.path.join(args.output_path, name))
                tifffile.imsave(os.path.join(args.output_path, name), img, dtype=np.float64)

if __name__ == "main":
    main()