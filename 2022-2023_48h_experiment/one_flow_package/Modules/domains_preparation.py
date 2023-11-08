# Module

# ---------------------------------------------------------------------- #
# Name :        domains_preparation.py
# Purpose :     Prepare nansat objecys and define domains  
# ---------------------------------------------------------------------- #

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from osgeo import gdal
from osgeo import osr
from nansat import Nansat, Domain, NSR

from sea_ice_drift import get_n
from sea_ice_drift.lib import get_spatial_mean, get_uint8_image

def prepare_nansat_objects(sar1, sar2, output_folder, polarisation):
    
    """
    Prepare and plot Nansat objects from given SAR images.
    
    The function creates Nansat objects out of a pair of SAR images and saves a side-by-side visualization of the
    processed images. The generated plots and images are saved to a specified output folder, organized by timestamps 
    and polarisation.

    Parameters:
    - sar1: The first SAR image object with attributes 'safe_filepath' and 'timestamp'.
    - sar2: The second SAR image object with attributes 'safe_filepath' and 'timestamp'.
    - polarisation (optional): The polarisation to be used, either 'HV' or 'HH'.

    Returns:
    - n1: Nansat object of the first SAR image after processing.
    - n2: Nansat object of the second SAR image after processing.
    
    """
    
    f1 = sar1.filepath  
    f2 = sar2.filepath
    
    
    n1 = get_n(f1, bandName= f'sigma0_{polarisation}', remove_spatial_mean=True) 
    n2 = get_n(f2, bandName= f'sigma0_{polarisation}', remove_spatial_mean=True)
    
    
    # Create directory for saving outputs for each pair of images
    output_dir_name = os.path.join(output_folder, f"{sar1.timestamp.strftime('%Y%m%dT%H%M%S')}_{sar2.timestamp.strftime('%Y%m%dT%H%M%S')}")
    try:
        os.makedirs(output_dir_name, exist_ok=True)
        print(f"Successfully created {output_dir_name}")
    except Exception as e:
        print(f"Failed to create {output_dir_name}. Error: {e}")
    
    # Create directory for saving plots
    plots_dir = os.path.join(output_dir_name, f"{polarisation}_plots")
    try:
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Successfully created {plots_dir}")
    except Exception as e:
        print(f"Failed to create {plots_dir}. Error: {e}")
    
    
    #Plot
    plt.close('all')
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    # Set background color to white
    ax[0].set_facecolor('white')
    ax[1].set_facecolor('white')
    fig.set_facecolor('white')
    
    im1 = ax[0].imshow(n1[1], clim=[0, 255])
    ax[0].axis('off')
    #plt.colorbar(im0, ax=ax[0])
    im2 = ax[1].imshow(n2[1], clim=[0, 255])
    ax[1].axis('off')
    #plt.colorbar(im1, ax=ax[1])
    cax = fig.add_axes([0.92, 0.13, 0.02, 0.75])  # Position and size of the colorbar
    cbar = fig.colorbar(im2, cax=cax)
    #plt.colorbar(im3, ax=ax[2])

    ax[0].set_title(f"SAR1 {polarisation} - {sar1.timestamp.strftime('%Y%m%dT%H%M%S')}")
    ax[1].set_title(f"SAR2 {polarisation} - {sar2.timestamp.strftime('%Y%m%dT%H%M%S')}")

    # Save the figure without displaying it
    save_path = os.path.join(plots_dir, f"n1_n2_{polarisation}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    # Close it after saving
    plt.close(fig)
  
    return n1, n2, output_dir_name, plots_dir

def prepare_grid(n1, n2, srs, X, Y, lon, lat, buffer):
    """
    Prepare a subset grid based on teh model grid and based on the bounds of the SAR image, expanded by a buffer.
    
    The function extracts a subset of the grid from the model based on the bounds 
    of the SAR image and adds a buffer to the subset. This buffered subset grid 
    is used for pattern matching to ensure that the drift output aligns seamlessly 
    with the model data.

    Parameters:
    - srs : Desired spatial reference system.
    - lon (xarray.DataArray): Longitudes from the model grid.
    - lat (xarray.DataArray): Latitudes from the model grid.
    - buffer (int): Number of pixels to expand the subset by.

    Returns:
    - X_subset (xarray.DataArray): Subset of X-coordinates based on the SAR image bounds (expanded by buffer).
    - Y_subset (xarray.DataArray): Subset of Y-coordinates based on the SAR image bounds (expanded by buffer).
    - lon_subset (xarray.DataArray): Subset of longitudes based on the SAR image bounds (expanded by buffer).
    - lat_subset (xarray.DataArray): Subset of latitudes based on the SAR image bounds (expanded by buffer).
    - lon1pm (array-like): Longitudes prepared for pattern matching.
    - lat1pm (array-like): Latitudes prepared for pattern matching.
    """
    # get lon/lat coordinates of the border of images
    lon1b, lat1b = n1.get_border()
    lon2b, lat2b = n2.get_border()

    #Get bound in rows and columns correspoding to the image borders set above
    r,c = np.where((lon.data > min(lon1b))*(lon.data <  max(lon1b))*(lat.data >  min(lat1b))*(lat.data < max(lat1b)))
    
    # Add buffer in pixels to make a subset biger to take into account drift
    min_row, max_row, min_col, max_col = min(r) - buffer, max(r) + buffer, min(c) - buffer, max(c) + buffer
    
    # Extract the subset grif out of the model grid based on the image bounds
    #That grid and srs will be used for pattern matching ensuring the drift output aligns easily with the model data.
    X_subset = X[min_col:max_col+1]
    Y_subset = Y[min_row:max_row+1]
    lon_subset = lon[min_row:max_row+1, min_col:max_col+1]
    lat_subset = lat[min_row:max_row+1, min_col:max_col+1]


    return X_subset, Y_subset, lon_subset, lat_subset

def plot_borders(mod_dom, n1, n2, output_dir_name):
    """
    Plot borders of model subset domain and two Sentinel images.

    Parameters:
    - mod_dom: Model subset domain object with a get_border() method
    - n1: First Sentinel image object with a get_border() method
    - n2: Second Sentinel image object with a get_border() method
    - output_dir_name: Directory path for saving the plotted figure

    Returns:
    - save_path: Full path to the saved figure
    """
    plt.close('all')
    # Set up the plot
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.plot(*mod_dom.get_border(), '.-', label='Model subset domain')
    axs.plot(*n1.get_border(), '.-', label='First S1 image')
    axs.plot(*n2.get_border(), '.-',  label='First S2 image')
    plt.legend()

    # Define save path
    general_save_path = os.path.join(output_dir_name, "General_plots")
    os.makedirs(general_save_path, exist_ok=True)
    save_path = os.path.join(general_save_path, "images_vs_domain_borders.png")
    # Save the figure
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close the figure
    plt.close(fig)

    return save_path
