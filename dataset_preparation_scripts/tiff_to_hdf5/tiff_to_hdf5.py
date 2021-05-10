
import h5py
import numpy as np
from PIL import Image 
import os 
import argparse 



def validatePath(path):
   if os.path.exists(path):
      return path
   raise Exception("Path is not valid")

parser = argparse.ArgumentParser()
parser.add_argument("--tiff-files-path", "-f", type=validatePath, required=True, 
                    help="Path to the folders where the tiff files are inside")

parser.add_argument("--output-absolute-path", "-o", required=True, type=str, 
                    help="absolut path of the created hdf5 file eg C:/test/out.hdf5")

args = parser.parse_args()


# default parameters of our CT-scan: 
detector_pixel_size = 0.000127 # [m]
distance_source_axis = 0.00363074699401855 # Source to Object-Stage (Center of rotation) [m]
distance_source_detector = 1.229268# [m]


def hdf5_tiff_builder(file_name: str, angle: int, detector_pixel_size: float,
                     distance_source_axis: float, distance_source_detector: float,
                    dimension, image, type_data): 
                     
   with h5py.File(file_name, "w") as f:
      angle = f.create_dataset("Angle", data=angle, dtype='float64')
      angle.attrs['MATLAB_class'] = 'double'
      f["DetectorPixelSizeX"] = detector_pixel_size
      f["DetectorPixelSizeX"].attrs['MATLAB_class'] = 'double'
      f["DetectorPixelSizeY"] = detector_pixel_size
      f["DetectorPixelSizeY"].attrs['MATLAB_class'] = 'double'
      
      dimension = f.create_dataset("Dimension", data=dimension, dtype='uint16')
      dimension.attrs['MATLAB_class'] = 'char'
      dimension.attrs['MATLAB_int_decode'] = 2
      

      f["DistanceSourceAxis"] = distance_source_axis
      f["DistanceSourceAxis"].attrs['MATLAB_class'] = 'double'
      f["DistanceSourceDetector"] = distance_source_detector
      f["DistanceSourceDetector"].attrs['MATLAB_class'] = 'double'
      image = f.create_dataset("Image", data=image, dtype='float64')
      image.attrs['MATLAB_class'] = 'double'    
      type_ = f.create_dataset("Type", data=type_data, dtype='uint16')
      type_.attrs['MATLAB_class'] = 'char'
      type_.attrs['MATLAB_int_decode'] = 2


def load_images_in_numpy_array(image_path): 
   images = []

   for root, dirs, files in os.walk(image_path):
      for name in sorted(files):
         try: 
            images.append(np.array(Image.open(os.path.join(root, name)), dtype=np.float64)) 
         except: 
            raise Exception("Folder does contain files which are no readable images")

   stacked_images = np.stack(images)
   return stacked_images



# extracted and saved them from the CT/Matlab_KSK/ReferenceProjections.hdf5 in ctutils 
dimension_data = np.loadtxt(os.path.join('parameter_files', 'dimension_hdf5.txt'), delimiter=',')
type_data = np.loadtxt(os.path.join('parameter_files', 'type_hdf5.txt'), delimiter=',')

images = load_images_in_numpy_array(args.tiff_files_path)


# angles are in rad. start at 0 and go to shortly before 6.28 .. (2*pi)
angle = np.linspace(0, 2*np.pi, images.shape[0], dtype=np.float64) 

hdf5_tiff_builder(args.output_absolute_path, angle, detector_pixel_size,
                   distance_source_axis, distance_source_detector, dimension_data,
                   images, type_data)

