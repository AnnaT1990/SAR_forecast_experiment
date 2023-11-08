# Module

# ---------------------------------------------------------------------- #
# Name :        config.py
# Purpose :     S1 Image class definitions and functions for preparing pairs of SAR1 and SAR2
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
# USER DEFINED PARAMETERS 
#
# ---------------------------------------------------------------------- #

import os
import numpy as np
import xarray as xr
import re
from nansat import Nansat, Domain, NSR

# -----------------------------
# Directories and File Paths
# -----------------------------

# Path to directory containing geocoded HH-polarized Sentinel-1 images
path_to_HH_files = '/home/jovyan/experiment_data/2022-2023_48h_experiment/SAR_images/HH_HV_160_for_batch_processing/test/HH_160'
#path_to_HH_files = '/home/jovyan/experiment_data/2022-2023_48h_experiment/SAR_images/HV_40'

# Path to directory containing geocoded HV-polarized Sentinel-1 images
path_to_HV_files = '/home/jovyan/experiment_data/2022-2023_48h_experiment/SAR_images/HH_HV_160_for_batch_processing/test/HV_160'
#path_to_HV_files = '/home/jovyan/experiment_data/2022-2023_48h_experiment/SAR_images/HV_40'

# Directory containing the SAFE formatted Sentinel-1 images
safe_folder = '/home/jovyan/experiment_data/2022-2023_48h_experiment/SAR_images/safe_test'
safe_folder = '/home/jovyan/experiment_data/2022-2023_48h_experiment/SAR_images/safe'

output_folder =  '/home/jovyan/experiment_data/2022-2023_48h_experiment/batch_output'

input_folder = '/home/jovyan/experiment_data/2022-2023_48h_experiment/one_flow_input'
# -----------------------------
# Regular Expressions
# -----------------------------

# Regular expression for matching geocoded Sentinel-1 product file names
# Such suffixes as Orb, Cal, TC, polarisation band and then resolution are added during geocoding based on steps and parameters
# Geocoding is happening in a separate environment using snappy module
S1_prod_regex = (r'S1[AB]{1}_EW_GRDM_1[A-Z]{3}_(?P<start_timestamp>[0-9]{8}T[0-9]{6})_'
                r'[0-9]{8}T[0-9]{6}_[0-9]{6}_[0-9A-Z]{6}_(?P<product_id>[0-9A-Z]{4})_Orb_Cal_TC_'
                r'(?P<polarisation>H[HV]{1})_(?P<resolution>[0-9]{1,4})')

# Regular expression for matching SAFE formatted Sentinel-1 file names
S1_safe_regex = (r'S1[AB]{1}_EW_GRDM_1[A-Z]{3}_(?P<start_timestamp>[0-9]{8}T[0-9]{6})_'
                r'[0-9]{8}T[0-9]{6}_[0-9]{6}_[0-9A-Z]{6}_(?P<product_id>[0-9A-Z]{4})')


# Compile the regular expression for matching Sentinel file names.
S1_prod_regex = re.compile(S1_prod_regex)  
S1_safe_regex = re.compile(S1_safe_regex) 


# -----------------------------
# Grid Configuration
# -----------------------------

# Load the data grid inforamtion extracted from BArents2.5 model 
grid_path_input = os.path.join(input_folder, "barent_grid.npz")
data = np.load(grid_path_input)


# Extract the X, Y, longitude, and latitude coordinates
# It will be a base for creating a model domain for retriving sar drift with the same resolution as Barents2.5 data 
# and for comparison domain where all SAR images (real and forecasted) will be projected to  
lon = data['lon']
lat = data['lat']
X = data['X']
Y = data['Y']
# Extract the proj4 string defining the Barents2.5's Lambert Conformal projection
proj4 = str(data['proj4'])

# Convert the proj4 string to a Nansat spatial reference object
srs = NSR(proj4)