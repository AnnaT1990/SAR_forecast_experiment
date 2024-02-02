# Module

# ---------------------------------------------------------------------- #
# Name :        s1_preparation.py
# Purpose :     S1 Image class definitions and functions for preparing pairs of SAR1 and SAR2
# ---------------------------------------------------------------------- #

import re
import datetime
import os


''' 
Comments
----------

SAFE files were geocoded separatly usign snappy module usifn following parameters:

sentinel_processor(in_folder, output_folder, polarisation_mode = 'DH', polarisation_bands = 'HH,HV', pixel_spacing = 160, crs = custom_crs_wkt, resampling_method = 'BILINEAR_INTERPOLATION')

Regular expression for output file differs from the original safe files by having a few extra suffixes at the end, for example:
S1A_EW_GRDM_1SDH_20221120T080155_20221120T080259_045975_05805E_51E1_Orb_Cal_TC_HV_160.tif

custom_crs_wkt = 'PROJCS["unknown",GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371000,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],
                PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",77.5],PARAMETER["central_meridian",-25],PARAMETER["standard_parallel_1",77.5],
                PARAMETER["standard_parallel_2",77.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],
                AXIS["Northing",NORTH]]'
'''

class S1Image:
    
    ''' 
    Container for Sentinel-1 product. 
    
    It contains information about path to SAFE file and timestamp.
    It also extract 2 filapaths to corresponding geocoded HH and HV tiff files matching them by similar product ID in regex patterns.
    
    This class holds information about the path to a SAFE file.
    It extracts its timestamp, product ID and file paths to the corresponding geocoded
    HH and HV TIFF files by matching them using similar product IDs defined in regex patterns.
    
    
    Attributes
    ----------
    filepath : str
        - Full path to the Sentinel-1 SAFE product file.
    filename : str
        - Name of the Sentinel-1 SAFE product file.
    timestamp : datetime.datetime
        - Acquisition time of the product.
    id : str
        - Sentinel-1 image ID.
    HH_tif_filepath : str
        - Full path to the corresponding HH polarisation TIFF file.
    HV_tif_filepath : str
        - Full path to the corresponding HV polarisation TIFF file.
        
    Class Methods
    -------------
    find_hh_tif_filepath(filename) : str
        - Locates and returns the path of the geocoded HH polarisation TIFF file 
          corresponding to the given SAFE file name.
    find_hv_tif_filepath(filename) : str
        - Locates and returns the path of the geocoded HV polarisation TIFF file 
          corresponding to the given SAFE file name.
          
    '''
   

    @classmethod
    def find_hh_tif_filepath(cls, filename, path_to_HH_files, S1_safe_regex, S1_prod_regex ):
        safe_match = S1_safe_regex.match(filename)
        if not safe_match:
            return None
        for tif_file in os.listdir(path_to_HH_files):
            tif_match = S1_prod_regex.match(tif_file)
            if tif_match and safe_match.group('product_id') == tif_match.group('product_id'):
                return os.path.join(path_to_HH_files, tif_file)
        return None

    @classmethod
    def find_hv_tif_filepath(cls, filename,  path_to_HV_files, S1_safe_regex, S1_prod_regex):
        safe_match = S1_safe_regex.match(filename)
        if not safe_match:
            return None
        for tif_file in os.listdir(path_to_HV_files):
            tif_match = S1_prod_regex.match(tif_file)
            if tif_match and safe_match.group('product_id') == tif_match.group('product_id'):
                return os.path.join(path_to_HV_files, tif_file)
        return None

    def __init__(self, filepath,  path_to_HH_files, path_to_HV_files, S1_safe_regex, S1_prod_regex):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        # Attempt to match the filename against the predefined regex pattern.
        match = S1_safe_regex.search(self.filename)
        if not match:
            raise ValueError(f"File {self.filename} does not match the expected pattern.")
        self.timestamp = datetime.datetime.strptime(match.group('start_timestamp'), '%Y%m%dT%H%M%S')
        self.id = match.group('product_id')
        self.HH_tif_filepath = self.find_hh_tif_filepath(self.filename, path_to_HH_files, S1_safe_regex, S1_prod_regex)
        self.HV_tif_filepath = self.find_hv_tif_filepath(self.filename, path_to_HV_files, S1_safe_regex, S1_prod_regex)
        self.abs_orbit = int(match.group('absolute_orbit'))
        self.mission_id = match.group('mission')
        if self.mission_id == "A":
            self.rel_orbit = (self.abs_orbit - 73) % 175 + 1
        elif self.mission_id == "B":
            self.rel_orbit = (self.abs_orbit - 27) % 175 + 1
        else:
            self.rel_orbit = None
        
         
        
    def __repr__(self):
        return f"S1Image(ID={self.id}, timestamp={self.timestamp.strftime('%Y%m%dT%H%M%S')}, relative_orbit = {self.rel_orbit})"

    
        
# Function to collect and create S1Image objects for all Sentinel files in a directory.
def collect_sentinel_files(directory, path_to_HH_files, path_to_HV_files, S1_safe_regex, S1_prod_regex):
    # Initialize an empty list to store S1Image objects
    sentinel_files = []

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .tif extension
        if filename.endswith('.SAFE'):
            # Create the full path to the file
            full_path = os.path.join(directory, filename)
            
            # Create a S1Image object and add it to the list
            try:
                sentinel_file = S1Image(full_path, path_to_HH_files, path_to_HV_files, S1_safe_regex, S1_prod_regex)
                sentinel_files.append(sentinel_file)
            except ValueError:
                # This will skip files that don't match the expected pattern
                continue

    # Check if the list is empty and raise an exception
    if not sentinel_files:
        raise ValueError("No objects were created as no files matched the criteria.")
        
    # Sort the list of S1Image objects by their timestamp
    sorted_files = sorted(sentinel_files, key=lambda x: x.timestamp)
    
    return sorted_files
    

# Function to get pairs of Sentinel files where the difference in their timestamps is within a given limit.
def get_pairs_within_time_limit(files, hours=60):
    pairs = []
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            if (files[j].timestamp - files[i].timestamp).total_seconds() <= hours * 3600:
                pairs.append((files[i], files[j]))
            else:
                break  # break inner loop since the list is sorted and no other pairs will fit the criteria

    if not pairs:
        print(f"No pairs found with a time difference of {hours} hours or less.")

    return pairs
