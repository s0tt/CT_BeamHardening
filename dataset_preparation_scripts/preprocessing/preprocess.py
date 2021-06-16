import numpy as np
import argparse
import tifffile
import matplotlib.pyplot as plt
import os

DATATYPE_PYTHON = "float32"
DATATYPE_NUMPY = np.float32

def ffc(img_ct, img_white):
    img_shape = np.array(img_ct).shape

    # take zeros as reference black image
    black = np.zeros((img_shape), dtype=DATATYPE_NUMPY)

    # take reference white image from aRTist
    white = np.array(img_white, dtype=DATATYPE_NUMPY)

    # flat-field-correction (ffc)
    img_new = ((img_ct - black) / (white-black))
    
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
        img_white = np.array(tifffile.imread(args.white_path), dtype=DATATYPE_NUMPY)
        for name in sorted(files):
            img = np.array(tifffile.imread(os.path.join(root, name)), dtype=DATATYPE_NUMPY)
            img = ffc(img, img_white)
            img = conv_attenuation(img)

            if args.overwrite == 1:
                #img_pil.save(os.path.join(root, name))
                tifffile.imsave(os.path.join(root, name), img, dtype=DATATYPE_NUMPY)
            else:
                #img_pil.save(os.path.join(args.output_path, name))
                tifffile.imsave(os.path.join(args.output_path, name), img, dtype=DATATYPE_NUMPY)

if __name__ == "__main__":
    main()