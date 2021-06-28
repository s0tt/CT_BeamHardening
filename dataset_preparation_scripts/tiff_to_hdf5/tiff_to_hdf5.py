
import h5py
import numpy as np
from PIL import Image 
import os 
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from dataset_preparation_scripts.preprocessing.preprocess import ffc, conv_attenuation

DATATYPE_PYTHON = "float32"
DATATYPE_NUMPY = np.float32

def validatePath(path):
   if os.path.exists(path):
      return path
   raise Exception("Path is not valid")

parser = argparse.ArgumentParser()
parser.add_argument("--tiff-files-path", "-f", type=validatePath, required=True, 
                    help="Path to the folders where the tiff files are inside")

parser.add_argument("--output-absolute-path", "-o", required=True, type=str, 
                    help="absolut path of the created hdf5 file eg C:/test/out.hdf5")

parser.add_argument("--white-path", "-w", required=False, type=str,
                    help="path of white detector reference .tiff file")
                    
parser.add_argument("--dist-source-rot-axis", "-dsra", required=True, type=float, 
                    help="distance between the source and the rotation axis of the object [m]")

parser.add_argument("--debug", "-dbg", required=False, type=int, default=0,
                    help="Debug mode, 0=Off 1=On")

args = parser.parse_args()


# default parameters of our CT-scan: 
detector_pixel_size = 0.000127 # [m]
distance_source_detector = 1.229268# [m]


def hdf5_tiff_builder(file_name: str,  detector_pixel_size: float,
                     distance_source_rot_axis: float, distance_source_detector: float,
                    dimension, image_paths, type_data): 
                     
   with h5py.File(file_name, "w") as f:
      # angles are in rad. start at 0 and go to shortly before 6.28 .. (2*pi)
      angle_rad = -1*np.arange(0, 2*np.pi, (2*np.pi)/5300, dtype=np.float64)
      angle = f.create_dataset("Angle", data=angle_rad, dtype='float64')
      angle.attrs['MATLAB_class'] = 'double'
      f["DetectorPixelSizeX"] = detector_pixel_size
      f["DetectorPixelSizeX"].attrs['MATLAB_class'] = 'double'
      f["DetectorPixelSizeY"] = detector_pixel_size
      f["DetectorPixelSizeY"].attrs['MATLAB_class'] = 'double'
      
      dimension = f.create_dataset("Dimension", data=dimension, dtype='uint16')
      dimension.attrs['MATLAB_class'] = 'char'
      dimension.attrs['MATLAB_int_decode'] = 2
      

      f["DistanceSourceAxis"] = distance_source_rot_axis
      f["DistanceSourceAxis"].attrs['MATLAB_class'] = 'double'
      f["DistanceSourceDetector"] = distance_source_detector
      f["DistanceSourceDetector"].attrs['MATLAB_class'] = 'double'

      image_dataset = None
      overall_border_vals = None
      for idx, img_path in enumerate(image_paths):
         if image_dataset is None:
            img = load_image_in_numpy_array(img_path)
            if args.debug:
               overall_border_vals = [np.amax(img), np.amin(img)]
            image_dataset = f.create_dataset("Image", data=img, dtype=DATATYPE_PYTHON, chunks=True, maxshape=(len(image_paths), 2304, 3200))
         else:
            img_current = load_image_in_numpy_array(img_path)
            if args.debug:
               max_val = np.amax(img_current)
               min_val = np.amin(img_current)
               print("Image:\t{}\tShape: {} \tMax: {}\t Min: {}".format(idx, img_current.shape, max_val, min_val))
               overall_border_vals[0] = max_val if max_val > overall_border_vals[0] else overall_border_vals[0]
               overall_border_vals[1] = min_val if min_val < overall_border_vals[1] else overall_border_vals[1]
            image_dataset.resize(image_dataset.shape[0]+img_current.shape[0], axis=0)
            image_dataset[idx, :, :] = img_current
      image_dataset.attrs['MATLAB_class'] = 'double'
      type_ = f.create_dataset("Type", data=type_data, dtype='uint16')
      type_.attrs['MATLAB_class'] = 'char'
      type_.attrs['MATLAB_int_decode'] = 2
      if args.debug:
         print("Image data finished:\tShape: {} \tMax: {}\t Min: {}".format(image_dataset.shape, overall_border_vals[0], overall_border_vals[1]))


def load_image_in_numpy_array(image_path): 
         try: 
            img = Image.open(image_path)
            img = preprocess(np.array(img, dtype=DATATYPE_NUMPY))
            return np.expand_dims(img, axis=0)
         except: 
            raise Exception("image path not readable")


def preprocess(img):
   if args.white_path is not None:
      img_white = np.array(Image.open(args.white_path), dtype=DATATYPE_NUMPY)
      img = ffc(img, img_white)
   else:
      print("Warning: White file not provided, FFC transform not executed")

   # cut small values
   img = np.maximum((1/60000), img)
   
   img = conv_attenuation(img)
   return img



def get_image_list(image_dir):
   img_list = []
   for root, dirs, files in os.walk(image_dir):
      for name in sorted(files):
         img_list.append(os.path.join(root, name))
   return img_list


# extracted and saved them from the CT/Matlab_KSK/ReferenceProjections.hdf5 in ctutils 
dimension_data = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                 'parameter_files', 'dimension_hdf5.txt'), delimiter=',')
type_data = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                 'parameter_files', 'type_hdf5.txt'), delimiter=',')

img_paths = get_image_list(args.tiff_files_path)

if args.debug:
   print("DBG Mode activated")
   print("Image List loaded, Length:", str(len(img_paths)))

hdf5_tiff_builder(args.output_absolute_path, detector_pixel_size,
                   args.dist_source_rot_axis, distance_source_detector, dimension_data,
                   img_paths, type_data)

