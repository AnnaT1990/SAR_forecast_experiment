# Module

# ---------------------------------------------------------------------- #
# Name :        SAR1_SAR2_drift_retrivial.py
# Purpose :     Retrieve reference drift for  further comparison with the model data and reference forecast.  
# ---------------------------------------------------------------------- #

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from osgeo import gdal
from osgeo import osr


from sea_ice_drift.lib import get_spatial_mean, get_uint8_image

from sea_ice_drift.ftlib import feature_tracking
#from sea_ice_drift.pmlib import pattern_matching

import sys
#sys.path.append(r'./sea_ice_drift') -> in fure rebuild docker image so it would add pmlib_with_ssim into sea_ice_drift module
#Then you can us: #from sea_ice_drift.pmlib_with_ssim import pattern_matching
sys.path.append(r'./modules/sea_ice_drift')
from pmlib_with_ssim import pattern_matching

def run_feature_tracking(n1, n2, plots_dir):  
    
    """
    Execute feature tracking between two SAR images and visualize the results.

    This function performs feature tracking between two Nansat objects representing SAR images. 
    It identifies and matches keypoints (features) between the images, then visualizes:
    1. The matched keypoints in geographic coordinates.
    2. The ice drift vectors superimposed on the first SAR image.

    Parameters:
    - n1: The first Nansat object representing a SAR image 1.
    - n2: The second Nansat object representing a SAR image 2.
    - plots_dir: Directory path where the visualizations will be saved.

    Returns:
    - c1, r1: Column and row coordinates of matched keypoints in the first image.
    - c2, r2: Column and row coordinates of matched keypoints in the second image.
    - lon1b, lat1b: Longitude and latitude coordinates of the border of the first image.

    """
        
    # get start/end coordinates in the image coordinate system (colums/rows)  
    c1, r1, c2, r2 = feature_tracking(n1, n2, nFeatures=50000, ratio_test=0.6, max_drift=100000, verbose=True)

    # Plot identified and matched keypoints in geographic coordinates

    # convert row/column coordinates of matched features to lon/lat
    lon1ft, lat1ft = n1.transform_points(c1, r1)
    lon2ft, lat2ft = n2.transform_points(c2, r2)

    # get lon/lat coordinates of the border of images
    lon1b, lat1b = n1.get_border()
    lon2b, lat2b = n2.get_border()


    # get hi-res landmask
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'])

    fig, ax = plt.subplots(1,2, figsize=(10,10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70))
    ax.add_feature(land_50m, zorder=0, edgecolor='black')
    ax.plot(lon1ft, lat1ft, '.', label='keypoints_1', transform=ccrs.PlateCarree())
    ax.plot(lon2ft, lat2ft, '.', label='keypoints_2', transform=ccrs.PlateCarree())
    ax.plot(lon1b, lat1b, '.-', label='border_1', transform=ccrs.PlateCarree())
    ax.plot(lon2b, lat2b, '.-', label='border_2', transform=ccrs.PlateCarree())
    ax.legend()

    # Save the figure without displaying it
    save_path = os.path.join(plots_dir, f"ft_matching_points.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


    # Plot ice drift on top of image_1

    # end points in image_1 coordinate system
    n1c2, n1r2 = n1.transform_points(lon2ft, lat2ft, DstToSrc=1)

    # ice drift components in image_1 coordinate system
    dr = n1r2 - r1
    dc = n1c2 - c1

    # border of image_2 in image_1 coordinate system
    n1lon2b, n1lat2b = n1.transform_points(lon2b, lat2b, DstToSrc=1)

    # plot of ice drift.
    fig = plt.figure(figsize=(10,10))
    plt.imshow(n1[1], cmap='gray')
    plt.quiver(c1, r1, dc, dr, color='r', angles='xy', scale_units='xy', scale=0.5)
    plt.plot(n1lon2b, n1lat2b, 'k.-')

    # Save the figure without displaying it
    save_path = os.path.join(plots_dir, f"ft_drift_vectors.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return c1, r1, c2, r2



def run_pattern_matching(plots_dir, x, y, lon1pm, lat1pm, n1, c1, r1, n2, c2, r2, srs, **kwargs):
    
    upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm = pattern_matching(lon1pm, lat1pm, n1, c1, r1, n2, c2, r2,
                                                          srs=srs.ExportToProj4(), **kwargs)


    # Plot main parameters
    
    plt.close('all')
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    
    # Update titles list with the new plot title
    titles = ['Displacement X', 'Displacement Y', 'Rotation', 'Correlation', 'Hessian', 'SSIM']
    # Flatten the ax array for easy indexing
    ax = ax.flatten()

    # Update the loop to include the new ma-trix and title
    for i, a in enumerate([upm, vpm, apm, rpm, hpm, ssim]):
        ax[i].set_title(titles[i])
        ax[i].set_facecolor('white')
        im = ax[i].imshow(a, extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(im, ax=ax[i])
        ax[i].set_xlim([x.min()-10000, x.max()-210000])
        ax[i].set_ylim([y.min()+110000, y.max()-160000])

        # Only show every second x-tick
        xticks = ax[i].get_xticks()
        ax[i].set_xticks(xticks[::2])  # Start to end, every second tick
    
    plt.tight_layout()
    fig.set_facecolor('white')
    #plt.show()
    # Save the figure without displaying it
    save_path = os.path.join(plots_dir, f"Pattern_matching_output_parameters.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    
    return upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm 

def combine_hh_hv(output_dir_name, x, y, upm_hh, vpm_hh, apm_hh, rpm_hh, hpm_hh, ssim_hh, lon2pm_hh, lat2pm_hh,
                  upm_hv, vpm_hv, apm_hv, rpm_hv, hpm_hv, ssim_hv, lon2pm_hv, lat2pm_hv):
    """
    This function combines the hv data with hh data where quality of hh data are higher
    (e.g. hessian hh > hessian hv),
    and then creates two plots to visualize the combined data.
    
    Parameters:
    - output_dir_name: Directory to save the generated plots
    - upm_hh, vpm_hh, apm_hh, ... : Data arrays for 'hh' output
    - upm_hv, vpm_hv, apm_hv, ... : Data arrays for 'hv' output
    
    Returns:
    - upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm : Combined data arrays
    """
    
    # Create a mask where hpm_hh is greater than hpm
    mask = hpm_hh > hpm_hv
    
    # Create new arrays as copies of the original arrays
    upm = upm_hv.copy()
    vpm = vpm_hv.copy()
    apm = apm_hv.copy()
    rpm = rpm_hv.copy()
    hpm = hpm_hv.copy()
    ssim = ssim_hv.copy()
    lon2pm = lon2pm_hv.copy()
    lat2pm = lat2pm_hv.copy()

    # Update the combined arrays using the mask
    upm[mask] = upm_hh[mask]
    vpm[mask] = vpm_hh[mask]
    apm[mask] = apm_hh[mask]
    rpm[mask] = rpm_hh[mask]
    hpm[mask] = hpm_hh[mask]
    ssim[mask] = ssim_hh[mask]
    lon2pm[mask] = lon2pm_hh[mask] 
    lat2pm[mask] = lat2pm_hh[mask]

    # Plotting1
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(hpm.flatten(), bins=50, alpha=0.5, label='Hessian HH + HV')
    ax.hist(hpm_hh.flatten(), bins=50, alpha=0.5, label='Hessian HH')
    ax.hist(hpm_hv.flatten(), bins=50, alpha=0.5, label='Hessian HV')
    

    ax.legend(loc='upper right')
    ax.set_title('Histogram of hessian HV, HH, and combined HH + HV')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    
    # Set background color to white
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    # Define save path
    general_save_path = os.path.join(output_dir_name, "General_plots")
    os.makedirs(general_save_path, exist_ok=True)
    save_path = os.path.join(general_save_path, "hessian_HH_vs_HV_histogram.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plotting2
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Scatter plot
    ax.scatter(hpm_hv.flatten(), hpm_hh.flatten(), alpha=0.5)

    # Add y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, label='y=x line')

    ax.set_title('Scatter plot of hessian HV vs hessian HH')
    ax.set_xlabel('Hessian HV')
    ax.set_ylabel('Hessian HH')
    ax.grid(True)
    ax.legend()
    
    # Set background color to white
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    # Define save path
    save_path = os.path.join(general_save_path, "hessian_HH_vs_HV_graph.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plotting3
    diff = hpm_hh-hpm_hv

    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    h_img = ax.imshow(diff, extent=[x.min(), x.max(), y.min(), y.max()], cmap='jet', alpha=1)
    ax.set_title('Areas where hessian for HH is higher than for HV')
    ax.set_xlim([x.min()-10000, x.max()-210000])
    ax.set_ylim([y.min()+110000, y.max()-160000])
    plt.colorbar(h_img, ax=ax, shrink=0.5)

    plt.tight_layout()
    # Set background color to white
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    # Save the figure without displaying it
    save_path = os.path.join(general_save_path, "hessian_HH_vs_HV_map.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
   
    
    return upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm

from scipy.ndimage import convolve

def get_good_pixel_indices(hpm, h_threshold, neighbors_threshold):
    """
    Get good pixel indices based on hessian and neighbor thresholds.

    Parameters:
    - hpm: Hessian processed matrix
    - h_threshold: Threshold for the hessian value
    - neighbors_threshold: Threshold for the number of valid neighboring pixels

    Returns:
    - gpi1: Good pixel index based on hessian value
    - gpi2: Good pixel index combining hessian and neighbors count
    """
    
    # Filtering arrays with hessian first, then excluding pixels with no neighbors
    filtered_hpm = np.where(hpm > h_threshold, hpm, np.nan)
    
    # Define a kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Convert nan values to 1 and valid values to 0
    nan_mask = np.isnan(filtered_hpm).astype(int)

    # Count nan neighbors
    nan_neighbors = convolve(nan_mask, kernel, mode='constant')

    # Count valid neighbors by subtracting nan neighbors from total neighbors (8 for a 3x3 kernel)
    valid_neighbors = 8 - nan_neighbors

    # Mask out pixels with zero valid neighbors
    filtered_hpm[valid_neighbors < neighbors_threshold] = np.nan

    # Filter vectors with hessian value
    gpi1 = (hpm > h_threshold)
    gpi2 = (hpm > h_threshold) & (valid_neighbors >= neighbors_threshold)
    
    return gpi1, gpi2

