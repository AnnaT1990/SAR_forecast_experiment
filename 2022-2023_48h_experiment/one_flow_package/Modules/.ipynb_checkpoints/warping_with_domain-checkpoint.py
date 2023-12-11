# Module


# ---------------------------------------------------------------------- #
# Name :        warping_with_domain.py
# Purpose :     Run warping using different domains 
# ---------------------------------------------------------------------- #

import matplotlib.pyplot as plt
from nansat import Nansat, Domain, NSR
import numpy as np
from scipy.interpolate import LinearNDInterpolator
#from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_edt


def get_dst_rows_cols(dst_dom):
    """ Create grids with row, column coordinates of the destination domain """
    rows2, cols2 = np.meshgrid(
        np.arange(0, dst_dom.shape()[0]),
        np.arange(0, dst_dom.shape()[1]),
        indexing='ij',
    )
    return rows2, cols2

def warp_with_rowcol(src_dom, src_img, c1, r1, c2, r2, dst_dom):
    """ Train interpolators of coordinates and apply to full resolution coordinates to computed a warped image """
    interp_r1 = LinearNDInterpolator(list(zip(r2, c2)), r1)
    interp_c1 = LinearNDInterpolator(list(zip(r2, c2)), c1)
    rows2, cols2 = get_dst_rows_cols(dst_dom)
    r1a = np.clip(interp_r1((rows2, cols2)), 0, src_dom.shape()[0])
    c1a = np.clip(interp_c1((rows2, cols2)), 0, src_dom.shape()[1])
    dst_img = map_coordinates(src_img, (r1a, c1a), order=0)
    return dst_img

def warp_distance(dst_dom, lon1, lat1, mask):
    """ Create a matrix with distance to the nearest valid drift and warp it onto the destination domain """
    c2_dist, r2_dist = dst_dom.transform_points(lon1.flatten(), lat1.flatten(), DstToSrc=1)
    mask_dist = distance_transform_edt(mask)
    interp_dist = LinearNDInterpolator(list(zip(r2_dist, c2_dist)), mask_dist.flatten())
    rows2, cols2 = get_dst_rows_cols(dst_dom)
    dst_dist = interp_dist((rows2, cols2))
    return dst_dist

def warp_with_lonlat(src_dom, src_img, lon1, lat1, lon2, lat2, dst_dom):
    """ Warp input image on destination domain if vectors of lon,lat source and destination points are knwown """
    c1, r1 = src_dom.transform_points(lon1.flatten(), lat1.flatten(), DstToSrc=1)
    c2, r2 = dst_dom.transform_points(lon2.flatten(), lat2.flatten(), DstToSrc=1)
    dst_img = warp_with_rowcol(src_dom, src_img, c1, r1, c2, r2, dst_dom)
    return dst_img

def warp(src_dom, src_img, dst_dom, step=None):
    """ Warp input image on destination domain (without drift compensation) """
    if step is None:
        step = int(src_dom.shape()[0]/100)
    src_lon, src_lat = src_dom.get_geolocation_grids(step)
    dst_img = warp_with_lonlat(src_dom, src_img, src_lon, src_lat, src_lon, src_lat, dst_dom)
    return dst_img

def warp_and_mask_with_lonlat(src_dom, src_img, lon1, lat1, lon2, lat2, mask, dst_dom, max_dist=2, fill_value=0):
    """ Warp input image on destination domain with drift compensation and masking if lon,lat,mask matrices are given """
    lon1v, lat1v, lon2v, lat2v = [i[~mask] for i in [lon1, lat1, lon2, lat2]]
    src_img = src_img.astype(float)
    dst_img = warp_with_lonlat(src_dom, src_img, lon1v, lat1v, lon2v, lat2v, dst_dom)
    dst_dist = warp_distance(dst_dom, lon1, lat1, mask)
    dst_img[(dst_dist > max_dist) + np.isnan(dst_dist)] = fill_value
    return dst_img


def warp_with_uv(src_dom, src_img, uv_dom, u, v, mask, dst_dom):
    """ Warp input image on destination domain with drift compensation and masking if U,V,mask matrices are given """
    uv_srs = NSR(uv_dom.vrt.get_projection()[0])
    lon1uv, lat1uv = uv_dom.get_geolocation_grids()
    x1, y1, _ = uv_dom.vrt.transform_coordinates(NSR(), (lon1uv[~mask], lat1uv[~mask]), uv_srs)
    x2 = x1 + u[~mask]
    y2 = y1 + v[~mask]
    lon2uv, lat2uv, _ = uv_dom.vrt.transform_coordinates(uv_srs, (x2, y2), NSR())
    inp_img = np.array(src_img)
    inp_img[0] = 0
    inp_img[-1] = 0
    inp_img[:, 0] = 0
    inp_img[:, -1] = 0
    dst_img = warp_with_lonlat(src_dom, inp_img, lon1uv[~mask], lat1uv[~mask], lon2uv, lat2uv, dst_dom)
    return dst_img

def warp_and_mask_with_uv(src_dom, src_img, uv_dom, u, v, mask, dst_dom, max_dist=2, fill_value=0):
    """ Warp input image on destination domain with drift compensation and masking if U,V,mask matrices are given """
    uv_srs = NSR(uv_dom.vrt.get_projection()[0])
    lon1uv, lat1uv = uv_dom.get_geolocation_grids()
    x1, y1, _ = uv_dom.vrt.transform_coordinates(NSR(), (lon1uv, lat1uv), uv_srs)
    x2 = x1 + u
    y2 = y1 + v
    lon2uv, lat2uv, _ = uv_dom.vrt.transform_coordinates(uv_srs, (x2, y2), NSR())
    inp_img = np.array(src_img)
    inp_img[0] = 0
    inp_img[-1] = 0
    inp_img[:, 0] = 0
    inp_img[:, -1] = 0
    dst_img = warp_and_mask_with_lonlat(src_dom, inp_img, lon1uv, lat1uv, lon2uv, lat2uv, mask, dst_dom, max_dist=max_dist, fill_value=fill_value) #np.nan if want Nan
    return dst_img

# Functions for plotting warping results
import os
import numpy as np
import matplotlib.pyplot as plt
import gc


def normalize(array):
    """Normalize an array to the range [0, 1]."""
    array_min = array.min()
    array_max = array.max()
    return (array - array_min) / (array_max - array_min)

def gamma_correction(image, gamma):
    """Apply gamma correction to an image."""
    return image ** (1.0 / gamma)

#@profile
def plot_sar_forecast_images(general_save_path, file_name, s1_dst_dom_hv, s2_dst_dom_hv, s1_dst_dom_S_hv, s1_dst_dom_hh, s2_dst_dom_hh, s1_dst_dom_S_hh, gamma_value=1.2):
    """
    Plot and save SAR forecast images.

    Parameters:
    - general_save_path: Path to save the images.
    - file_name: The base name for the saved files.
    - s1_dst_dom_hv, s2_dst_dom_hv, s1_dst_dom_S_hv, s1_dst_dom_hh, s2_dst_dom_hh, s1_dst_dom_S_hh: Arrays representing different channels of SAR data for SAR1, SAR2 and SAR2_forecasted.
    - gamma_value: The gamma correction value; defaults to 1.2.
    """
    # Normalize and apply gamma correction
    s1_hv_gamma_corrected = gamma_correction(normalize(s1_dst_dom_hv), gamma_value)
    s2_hv_gamma_corrected = gamma_correction(normalize(s2_dst_dom_hv), gamma_value)
    s1_predicted_hv_gamma_corrected = gamma_correction(normalize(s1_dst_dom_S_hv), gamma_value)
    s1_hh_gamma_corrected = gamma_correction(normalize(s1_dst_dom_hh), gamma_value)
    s2_hh_gamma_corrected = gamma_correction(normalize(s2_dst_dom_hh), gamma_value)
    s1_predicted_hh_gamma_corrected = gamma_correction(normalize(s1_dst_dom_S_hh), gamma_value)

    # Create an empty blue channel
    blue_channel1 = np.zeros_like(s1_hv_gamma_corrected)
    blue_channel2 = np.zeros_like(s2_hv_gamma_corrected)
    blue_channel3 = np.zeros_like(s1_predicted_hv_gamma_corrected)

    # Stack the channels to make composite RGB images
    rgb_image1 = np.stack([s1_hv_gamma_corrected, s1_hh_gamma_corrected, blue_channel1], axis=-1)
    rgb_image2 = np.stack([s2_hv_gamma_corrected, s2_hh_gamma_corrected, blue_channel2], axis=-1)
    rgb_image3 = np.stack([s1_predicted_hv_gamma_corrected, s1_predicted_hh_gamma_corrected, blue_channel3], axis=-1)

    # Memory cleanup
    del s1_hv_gamma_corrected, s2_hv_gamma_corrected, s1_predicted_hv_gamma_corrected
    del s1_hh_gamma_corrected, s2_hh_gamma_corrected, s1_predicted_hh_gamma_corrected
    del blue_channel1, blue_channel2, blue_channel3
    gc.collect()

    # Display the composite images
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(rgb_image1)
    axs[1].imshow(rgb_image3)
    axs[2].imshow(rgb_image2)
    axs[0].set_title('SAR1')
    axs[1].set_title('SAR2 Predicted')
    axs[2].set_title('SAR2')

    # Set common limits for all subplots and background color to white
    for ax in axs:
        ax.set_xlim([0, 3200])
        ax.set_ylim([6100, 1700])
        ax.set_facecolor('white')

    fig.set_facecolor('white')
    plt.tight_layout()
    #plt.show()

    # Save the figure
    save_path = os.path.join(general_save_path, f"{file_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    del rgb_image1, rgb_image2, rgb_image3
    gc.collect()
    plt.close(fig)
