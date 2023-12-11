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

from scipy.ndimage import convolve


def run_feature_tracking(n1, n2, plots_dir, FT=True):  
    
    """
    Execute feature tracking between two SAR images and visualize the results.

    Parameters:
    - n1: The first Nansat object representing a SAR image 1.
    - n2: The second Nansat object representing a SAR image 2.
    - plots_dir: Directory path where the visualizations will be saved.
    - FT: Boolean, if True use feature tracking, otherwise use meshgrid approach.

    Returns:
    - c1, r1: Column and row coordinates of matched keypoints in the first image.
    - c2, r2: Column and row coordinates of matched keypoints in the second image.
    - lon1b, lat1b: Longitude and latitude coordinates of the border of the first image.
    """

    if FT:
        # Feature Tracking Approach
        c1, r1, c2, r2 = feature_tracking(n1, n2, nFeatures=50000, ratio_test=0.6, max_drift=100000, verbose=True)
    else:
        # Meshgrid Approach
        c1, r1 = np.meshgrid(
            np.arange(1, n1.shape()[1], 250),
            np.arange(1, n1.shape()[0], 250))
        c1, r1 = c1.flatten(), r1.flatten()
        lon1ft, lat1ft = n1.transform_points(c1, r1)
        c2, r2 = n2.transform_points(lon1ft, lat1ft, DstToSrc=True)

        margin = 200
        gpi = ((c1 > margin) * (r1 > margin) * (c1 < (n1.shape()[1]-margin)) * (r1 < (n1.shape()[0]-margin)) *
               (c2 > margin) * (r2 > margin) * (c2 < (n2.shape()[1]-margin)) * (r2 < (n2.shape()[0]-margin)))

        c1, r1, c2, r2 = c1[gpi], r1[gpi], c2[gpi], r2[gpi]
        
    

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
    
    plt.close('all')
    
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


'''
#@profile
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
'''

#@profile
def run_pattern_matching(plots_dir, x, y, lon1pm, lat1pm, n1, c1, r1, n2, c2, r2, srs, **kwargs):
    # Assume pattern_matching is defined elsewhere and returns the necessary values
    upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm = pattern_matching(lon1pm, lat1pm, n1, c1, r1, n2, c2, r2,
                                                                     srs=srs.ExportToProj4(), **kwargs)

    # Plot main parameters
    plt.close('all')
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    titles = ['Displacement X', 'Displacement Y', 'Rotation', 'Correlation', 'Hessian', 'SSIM']
    ax = ax.flatten()

    for i, matrix in enumerate([upm, vpm, apm, rpm, hpm, ssim]):
        ax[i].set_title(titles[i])
        ax[i].set_facecolor('white')
        im = ax[i].imshow(matrix, extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(im, ax=ax[i])
        ax[i].set_xlim([x.min()-10000, x.max()-210000])
        ax[i].set_ylim([y.min()+110000, y.max()-160000])
        xticks = ax[i].get_xticks()
        ax[i].set_xticks(xticks[::2])
    
    plt.tight_layout()
    fig.set_facecolor('white')

    # Save the combined figure
    save_path = os.path.join(plots_dir, "Pattern_matching_output_parameters.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save separate plots for Correlation, Hessian, and SSIM
    for i, matrix in enumerate([hpm, ssim]):
        plt.figure(figsize=(8, 8))
        plt.title(titles[i+4])  # 4 is the starting index for Correlation, Hessian, and SSIM in titles
        plt.imshow(matrix, extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar()
        plt.xlim([x.min()-10000, x.max()-210000])
        plt.ylim([y.min()+110000, y.max()-160000])
        plt.tight_layout()
        fig.set_facecolor('white')
        single_save_path = os.path.join(plots_dir, f"{titles[i+4].replace(' ', '_')}_plot.png")
        plt.savefig(single_save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm
    

def combine_based_on_hessian(output_dir_name, folder_name, x, y, upm_hh, vpm_hh, apm_hh, rpm_hh, hpm_hh, ssim_hh, lon2pm_hh, lat2pm_hh,
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
    general_save_path = os.path.join(output_dir_name, folder_name)
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
    
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    titles = ['Displacement X', 'Displacement Y', 'Rotation', 'Correlation', 'Hessian', 'SSIM']
    ax = ax.flatten()
    
    for i, matrix in enumerate([upm, vpm, apm, rpm, hpm, ssim]):
        ax[i].set_title(titles[i])
        ax[i].set_facecolor('white')
        im = ax[i].imshow(matrix, extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(im, ax=ax[i])
        ax[i].set_xlim([x.min()-10000, x.max()-210000])
        ax[i].set_ylim([y.min()+110000, y.max()-160000])
        xticks = ax[i].get_xticks()
        ax[i].set_xticks(xticks[::2])
    
    plt.tight_layout()
    fig.set_facecolor('white')

    # Save the combined figure
    save_path = os.path.join(general_save_path, "Pattern_matching_output_parameters_combined.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save separate plots for Correlation, Hessian, and SSIM
    for i, matrix in enumerate([hpm, ssim]):
        plt.figure(figsize=(8, 8))
        plt.title(titles[i+4])  # 4 is the starting index for Correlation, Hessian, and SSIM in titles
        plt.imshow(matrix, extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar()
        plt.xlim([x.min()-10000, x.max()-210000])
        plt.ylim([y.min()+110000, y.max()-160000])
        plt.tight_layout()
        fig.set_facecolor('white')
        single_save_path = os.path.join(general_save_path, f"{titles[i+4].replace(' ', '_')}_combined_plot.png")
        plt.savefig(single_save_path, dpi=300, bbox_inches='tight')
        plt.close()
   
    
    return upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm



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

def plot_filter_results(drift_save_path, x, y, hpm, upm, vpm, gpi1, gpi2, disp_legend_min, disp_legend_max, hessian, neighbors):
    
    u = upm/1000 
    v = vpm/1000 
    disp = np.sqrt((v**2+u**2))

    disp = np.where(np.isinf(disp), np.nan, disp)

    # Find global min and max displacement for consistent color range in all plots
    disp_min = np.nanmin(disp)
    disp_max = np.nanmax(disp)
    

    plt.close('all')
    fig, axs = plt.subplots(1,3, figsize=(30,10)) 

    gpi = (hpm>0)

    quiv_params = {
        'scale': 900,
        'cmap': 'jet',
        'width': 0.002,
        'headwidth': 3,
        'clim': (disp_legend_min, disp_legend_max) 
    }

    # Create quiver plots   
    quiv1 = axs[0].quiver(x[gpi][::1], y[gpi][::1], u[gpi][::1], v[gpi][::1], disp[gpi][::1],  **quiv_params)
    quiv2 = axs[1].quiver(x[gpi1][::1], y[gpi1][::1], u[gpi1][::1], v[gpi1][::1], disp[gpi1][::1],  **quiv_params)
    quiv3 = axs[2].quiver(x[gpi2][::1], y[gpi2][::1], u[gpi2][::1], v[gpi2][::1], disp[gpi2][::1],  **quiv_params)


    # Colorbar with shared color scale
    cbar = fig.colorbar(quiv1, ax=axs, orientation='vertical', shrink=0.9)
    cbar.set_label('Displacement Magnitude [km]')

    # Set the same x and y limits for all axes
    for ax in axs:
        ax.set_xlim([300000, 600000])
        ax.set_ylim([150000, 600000])

    axs[0].set_title(f'Ice Drift Dispalcement before filtering [km]\n{np.sum(gpi.data)} values')
    axs[1].set_title(f'Ice Drift Displacement with hessian > {hessian} [km]\n{np.sum(gpi1.data)} values')
    axs[2].set_title(f'Ice Drift Displacement with hessian > {hessian} and neighbors > {neighbors} [km]\n{np.sum(gpi2.data)} values')


    # Set background color to white
    #ax.set_facecolor('white')plot_filter_results
    fig.set_facecolor('white')

    plt.tight_layout
    
    plt.tight_layout

    # Save the figure without displaying it
    save_path = os.path.join(drift_save_path, f"Filtering_results_h{hessian}_n{neighbors}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return disp_min, disp_max

def save_results_to_npz(output_dir, save_name, **kwargs):
    
    # Define the path for the .npz file
    save_path = os.path.join(output_dir, f"{save_name}.npz")
    
    # Save the arrays into the .npz file
    np.savez(save_path, **kwargs)
    
    print(f"Arrays saved to {save_path}")
    
    return 


    
    