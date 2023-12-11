import sys
sys.path.append("./modules")



# Import general modules
from nansat import Nansat, Domain, NSR
import os 
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

# Import temporal modules needed for testing plotting
import matplotlib.pyplot as plt
#%matplotlib inline

# Import modules for calculate cumulative drift with hourly model data
# Using passive trackers method
from scipy.interpolate import RegularGridInterpolator



# Import SAR forecasting modules
import config
import s1_preparation
import domains_preparation
import SAR1_SAR2_drift_retrieval
import warping_with_domain
import model_data_processing

# Import variables
from config import path_to_HH_files, path_to_HV_files, safe_folder 
from config import output_folder, input_folder
from config import S1_prod_regex, S1_safe_regex
from config import lon, lat, X, Y, proj4, srs
from config import hessian, neighbors
from config import disp_legend_min
from config import disp_legend_max

# For cleaning up memory
import gc

import cProfile
import pstats
from io import StringIO

# For ignoring some warnings
import warnings
# Ignore the runtime warnings related to 'Mean of empty slice' wen calculate rolling average
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

#======================================================================
# 1. Prepare SAR pairs
#----------------------------------------------------------------------

# Collect Sentinel SAFE objects for files in safe directory.
safe_objects = s1_preparation.collect_sentinel_files(safe_folder, path_to_HH_files, path_to_HV_files,  S1_safe_regex, S1_prod_regex)

# Get pairs of Sentinel SAFE objects where their timestamps are within 50 hours of each other.
sar_pairs = s1_preparation.get_pairs_within_time_limit(safe_objects, hours = 50)

# Print details for each pair.
for index, pair in enumerate(sar_pairs, start=1):  # start=1 makes the index start from 1
    print(f'Pair {index}:')
    print(f'SAR1: {pair[0].filename} \ntimestamp: {pair[0].timestamp}\n'
          f'SAR2: {pair[1].filename} \ntimestamp: {pair[1].timestamp}')

#======================================================================
# Loop through pairs for processing
#----------------------------------------------------------------------

# Define mod_dom and dst_dom outside the loop
# Filled up with the first pair domain and stay the same fro all other pairs
# So should be changed if pairs for different areas used
mod_res = 2500
mod_dom = None  # Initialize to None
dst_res = 100
dst_dom = None  # Initialize to None

#Initialize a JohannesroThredds object to interface with the specified THREDDS dataset.
jt = model_data_processing.JohannesroThredds()

# Loop over all pairs and use enumerate to get an index for each pair
for index, pair in enumerate(sar_pairs, start=1):  # start=1 to have human-friendly indexing
    
    # Create a Profile object
    pr = cProfile.Profile()
    pr.enable()  # Start profiling
    
    start_time = time.time()
    
    #======================================================================
    # 2.  Create nansat objects and define model and comparison domains
    #----------------------------------------------------------------------
    
    # Create directory for saving outputs for each pair of images (named after the timestamps of two images)
    output_dir_path = os.path.join(output_folder, f"{pair[0].timestamp.strftime('%Y%m%dT%H%M%S')}_{pair[1].timestamp.strftime('%Y%m%dT%H%M%S')}")
    try:
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Successfully created {output_dir_path}")
    except Exception as e:
        print(f"Failed to create {output_dir_path}. Error: {e}")
        
    # Create directory for saving output flots for FT and PM for SAR1-SAR2 drift retrivieal
    hh_hv_pm_plots_dir = os.path.join(output_dir_path,"hh_hv_pm_plots")
    try:
        os.makedirs(hh_hv_pm_plots_dir, exist_ok=True)
        print(f"Successfully created {hh_hv_pm_plots_dir}")
    except Exception as e:
        print(f"Failed to create {hh_hv_pm_plots_dir}. Error: {e}")
        
    # 2.1. Prepare nansat objects and domains for HV polarisation
    n1_hv, n2_hv, plots_dir_hv = domains_preparation.prepare_nansat_objects(
        pair[0], pair[1], hh_hv_pm_plots_dir, polarisation='HV')
    
    # Prepare nansat objects and domains for HH polarisation
    n1_hh, n2_hh, plots_dir_hh = domains_preparation.prepare_nansat_objects(
        pair[0], pair[1], hh_hv_pm_plots_dir, polarisation='HH')
    

    # 2.2  Define model domain (mod_dom) for comparing drift and comparison (dst_dom) domain to compare SAR images (real and forecasted)
 
    
    # Check if mod_dom and dst_dom are None, if so, define them based on the first pair
    if mod_dom is None:
        X_subset, Y_subset, lon_subset, lat_subset, min_row, max_row, min_col, max_col = domains_preparation.prepare_grid(n1_hv, n2_hv, srs, X, Y, lon, lat, buffer=0)
        mod_dom = Domain(srs, f'-te {min(X_subset.data)} {min(Y_subset.data) - mod_res * 2} {max(X_subset.data) + mod_res} {max(Y_subset.data)} -tr {mod_res} {mod_res}')
        lon1pm, lat1pm = mod_dom.get_geolocation_grids()
        x, y = mod_dom.get_geolocation_grids(dst_srs=srs)
        # Save domain grid variables the same for all pairs
        save_name = 'domain_output'
        sar_drift_output_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_folder, save_name,
                                                                         X=X_subset, Y=Y_subset)
    
    if dst_dom is None:
        dst_dom = Domain(srs, f'-te {min(X_subset.data)} {min(Y_subset.data) - dst_res * 2} {max(X_subset.data) + dst_res} {max(Y_subset.data)} -tr {dst_res} {dst_res}')
        
       
    '''
    # Pick this if you want to create mod and comparison domain for each pair (then they can differ)
    # Prepare subset model grid for domains and pattern matching
    X_subset, Y_subset, lon_subset, lat_subset, min_row, max_row, min_col, max_col = domains_preparation.prepare_grid(n1_hv, n2_hv, srs, X, Y, lon, lat, buffer=0)
    
    # Set a model domain
    mod_res = 2500
    mod_dom = Domain(srs, f'-te {min(X_subset.data)} {min(Y_subset.data) - mod_res * 2} {max(X_subset.data) + mod_res} {max(Y_subset.data)} -tr {mod_res} {mod_res}')
    
    
    lon1pm, lat1pm = mod_dom.get_geolocation_grids()
    x, y = mod_dom.get_geolocation_grids(dst_srs=srs)
    
    
    # Set a comparison domain 
    dst_res = 100
    dst_dom = Domain(srs, f'-te {min(X_subset.data)} {min(Y_subset.data) - dst_res * 2} {max(X_subset.data) + dst_res} {max(Y_subset.data)} -tr {dst_res} {dst_res}')
    
    # Save domain grid variables each own for every pair
    save_name = 'domain_output'
    sar_drift_output_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_dir_path, save_name,
                                                                         X=X_subset, Y=Y_subset)
                                                                         
    '''    
    
    # Create directory for saving plots related to drift vector fields (both SAR retrieved and model data)
    drift_plot_save_path = os.path.join(output_dir_path, "drift_vector_fields_plots")
    os.makedirs(drift_plot_save_path, exist_ok=True)
    
    domains_preparation.plot_borders(mod_dom, n1_hv, n2_hv, drift_plot_save_path) # borders for hh and hv are the same
    # Checking that domains have the same borders
    
    rows1, cols1 = dst_dom.shape()
    print("dst_dom corner coordinates:", dst_dom.transform_points([0,cols1-1,0,cols1-1], [0,0,rows1-1,rows1-1], dst_srs=srs))
    
    rows1, cols1 = mod_dom.shape()
    print("mod_dom corner coordinates:", mod_dom.transform_points([0,cols1-1,0,cols1-1], [0,0,rows1-1,rows1-1], dst_srs=srs))
    
    #print("1. Nansat objects created,  model and comparison domains defined.")
    #======================================================================
    # 3.   Retrieve SAR reference drift
    #----------------------------------------------------------------------
    
    # 3.1. Run feature tracking and pattern matching for HV
    
   
    # Run feature tracking and plot results  
    c1_hv, r1_hv, c2_hv, r2_hv = SAR1_SAR2_drift_retrieval.run_feature_tracking(n1_hv, n2_hv, plots_dir_hv, FT=True)
    # Check the number of keypoints
    if len(c1_hv) >= 4:
        print(f"Enough keypoints for PM -  {len(c1_hv)} > 4")
    else:
        print(f"Not enough key points for PM - {len(c1_hv)} < 4, skipping FT")
        c1_hv, r1_hv, c2_hv, r2_hv = SAR1_SAR2_drift_retrieval.run_feature_tracking(n1_hv, n2_hv, plots_dir_hv, FT= False)
    
    #Run pattern matching and plot results
    upm_hv, vpm_hv, apm_hv, rpm_hv, hpm_hv, ssim_hv, lon2pm_hv, lat2pm_hv = SAR1_SAR2_drift_retrieval.run_pattern_matching(plots_dir_hv, x, y, 
                                                               lon1pm, lat1pm, n1_hv, c1_hv, r1_hv, n2_hv, c2_hv, r2_hv, srs, 
                                                               #min_border=200,
                                                               #max_border=200,
                                                               #angles=[-50, -45, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 45, 50])
                                                               min_border=10, #test
                                                               max_border=10, #test
                                                               angles=[0]) #test
                                                               
    
    # 3.2. Run feature tracking and pattern matching for HH
    
    # HH Processing
    # Run feature tracking and plot results 
    c1_hh, r1_hh, c2_hh, r2_hh = SAR1_SAR2_drift_retrieval.run_feature_tracking(n1_hh, n2_hh, plots_dir_hh, FT=True)
    # Check the number of keypoints
    if len(c1_hh) >= 4:
        print(f"Enough keypoints for PM -  {len(c1_hh)} > 4")
    else:
        print(f"Not enough key points for PM - {len(c1_hh)} < 4, skipping FT")
        c1_hh, r1_hh, c2_hh, r2_hh = SAR1_SAR2_drift_retrieval.run_feature_tracking(n1_hh, n2_hh, plots_dir_hh, FT= False)
    
    #Run pattern matching and plot results
    upm_hh, vpm_hh, apm_hh, rpm_hh, hpm_hh, ssim_hh, lon2pm_hh, lat2pm_hh = SAR1_SAR2_drift_retrieval.run_pattern_matching(plots_dir_hh, x, y, 
                                                               lon1pm, lat1pm, n1_hh, c1_hh, r1_hh, n2_hh, c2_hh, r2_hh,srs, 
                                                               #min_border=200,
                                                               #max_border=200,
                                                               #angles=[-50, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 50 ])
                                                               min_border=10, #test
                                                               max_border=10, #test
                                                               angles=[0]) #test
                                                               
    
    
    # 3.3. Get combined drift and all textural parameters
    
    # Combining hh and hv results based on hessian threshold
    folder_name = "hh_hv_combined"
    upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm = SAR1_SAR2_drift_retrieval.combine_based_on_hessian(hh_hv_pm_plots_dir, folder_name, x, y, upm_hh, vpm_hh, apm_hh, rpm_hh, hpm_hh, ssim_hh, lon2pm_hh, lat2pm_hh, upm_hv, vpm_hv, apm_hv, rpm_hv, hpm_hv, ssim_hv, lon2pm_hv, lat2pm_hv)
   
    # 3.4.  Filter drift data with the good pixel indices based on hessian and neighbor thresholds.
    
    #Returns:
    #    - gpi1: Good pixel index based on hessian value
    #    - gpi2: Good pixel index combining hessian and neighbors count 

    # Calculate gpi mask
    gpi1, gpi2 = SAR1_SAR2_drift_retrieval.get_good_pixel_indices(hpm, h_threshold=hessian, neighbors_threshold=neighbors)
    
        
    # Plot the filtering results
    sar_disp_min, sar_disp_max = SAR1_SAR2_drift_retrieval.plot_filter_results(drift_plot_save_path, x, y, hpm, upm, vpm, gpi1, gpi2, disp_legend_min, disp_legend_max, hessian, neighbors)
    
    
    # Create directory for saving outputs for each pair of images (named after the timestamps of two images)
    output_data_dir = os.path.join(output_dir_path, "output_data")
    try:
        os.makedirs(output_data_dir, exist_ok=True)
        print(f"Successfully created {output_data_dir}")
    except Exception as e:
        print(f"Failed to create {output_data_dir}. Error: {e}")
    
    #  Save final reference drift, its parameters and filtering arrays to npy files
    save_name = 'sar_drift_output'
    SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name,
                                                                             upm=upm, vpm=vpm, apm=apm, rpm=rpm, 
                                                                             hpm=hpm, ssim=ssim, lon2pm=lon2pm, 
                                                                             lat2pm=lat2pm, gpi1=gpi1, gpi2=gpi2)
    
    #print("2. SAR reference drift Retrieved")
    #======================================================================
    # 4. Warp SAR1 image with the reference SAR drift and compare all SARs in the comparison domain
    #----------------------------------------------------------------------
    
    # 4.1. Warp
    # Warp SAR1 with SAR-drift compenstaion/displacement
    good_pixels = gpi2
    mask_pm = ~good_pixels # mask out low quality or NaN
    s1_dst_dom_S_hv = warping_with_domain.warp_with_uv(n1_hv, n1_hv[1], mod_dom, upm, vpm, mask_pm, dst_dom)
    s1_dst_dom_S_hh = warping_with_domain.warp_with_uv(n1_hh, n1_hh[1], mod_dom, upm, vpm, mask_pm, dst_dom)
    s1_dst_dom_S_hv_masked = warping_with_domain.warp_and_mask_with_uv(n1_hv, n1_hv[1], mod_dom, upm, vpm, mask_pm, dst_dom, max_dist=2, fill_value=0)
    s1_dst_dom_S_hh_masked = warping_with_domain.warp_and_mask_with_uv(n1_hh, n1_hh[1], mod_dom, upm, vpm, mask_pm, dst_dom, max_dist=2, fill_value=0)
    s1_dst_dom_S_hv_masked0 = warping_with_domain.warp_and_mask_with_uv(n1_hv, n1_hv[1], mod_dom, upm, vpm, mask_pm, dst_dom, max_dist=0, fill_value=0)
    s1_dst_dom_S_hh_masked0 = warping_with_domain.warp_and_mask_with_uv(n1_hh, n1_hh[1], mod_dom, upm, vpm, mask_pm, dst_dom, max_dist=0, fill_value=0)
    
    
    # Warp SAR2 to the comparison domain
    s2_dst_dom_hv = warping_with_domain.warp(n2_hv, n2_hv[1], dst_dom)
    s2_dst_dom_hh = warping_with_domain.warp(n2_hh, n2_hh[1], dst_dom)
    
    # Warp SAR1 to the comparison domain for visualisation
    s1_dst_dom_hv = warping_with_domain.warp(n1_hv, n1_hv[1], dst_dom)
    s1_dst_dom_hh = warping_with_domain.warp(n1_hh, n1_hh[1], dst_dom)
    

    # 4.2. Plot warping results
    warping_plots_save_dir = os.path.join(output_dir_path, "warped_arrays_plots")
    os.makedirs(warping_plots_save_dir, exist_ok=True)
    
    warping_with_domain.plot_sar_forecast_images(warping_plots_save_dir, 
                                                 "forecast_with_sar_ref_drift", 
                                                 s1_dst_dom_hv, s1_dst_dom_hh, 
                                                 s2_dst_dom_hv, s2_dst_dom_hh,
                                                 s1_dst_dom_S_hv, s1_dst_dom_S_hh,
                                                 gamma_value=1.2)
    
    warping_with_domain.plot_sar_forecast_images(warping_plots_save_dir, 
                                             "forecast_with_sar_ref_drift_masked_2dist", 
                                             s1_dst_dom_hv, s1_dst_dom_hh,
                                             s2_dst_dom_hv, s2_dst_dom_hh,
                                             s1_dst_dom_S_hv, s1_dst_dom_S_hh_masked,
                                             gamma_value=1.2)  #s1_dst_dom_S_hv_masked
    warping_with_domain.plot_sar_forecast_images(warping_plots_save_dir, 
                                         "forecast_with_sar_ref_drift_masked_0dist", 
                                         s1_dst_dom_hv, s1_dst_dom_hh,
                                         s2_dst_dom_hv, s2_dst_dom_hh,
                                         s1_dst_dom_S_hv,s1_dst_dom_S_hh_masked0,
                                         gamma_value=1.2) #s1_dst_dom_S_hv_masked0

    # 5.4. Save warped array to data output
    save_name = 'sar_warped_arrays'
    sar_distort_data_error_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name, 
                                                                                sar_warped_hv=s1_dst_dom_S_hv, sar_warped_hh=s1_dst_dom_S_hh,
                                                                                sar_warped_hv_masked2 = s1_dst_dom_S_hv_masked,
                                                                                sar_warped_hh_masked2 = s1_dst_dom_S_hh_masked,
                                                                                sar_warped_hv_masked0 = s1_dst_dom_S_hv_masked0, 
                                                                                sar_warped_hh_masked0 = s1_dst_dom_S_hh_masked0)
    
    # 5.5. Save SAR1 and SAR2 arrays warped to dst domain
    save_name = 'sar1_sar2_dst_domain'
    sar_distort_data_error_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name, 
                                                                                s2_dst_dom_hv=s2_dst_dom_hv,
                                                                                s2_dst_dom_hh=s2_dst_dom_hh,
                                                                                s1_dst_dom_hv = s1_dst_dom_hv,
                                                                                s1_dst_dom_hh = s1_dst_dom_hh)
    
    #print("3. Warped SAR1 image with the reference SAR drift.")
        
    #======================================================================
    # 5. Calculate  sar warping quality parametrs (corr, hess, ssim) for the predicted SAR2 (by calculating pattern matching on SAR2 and SAR2_predicted)
    #----------------------------------------------------------------------
    
    # 5.1. Make new nansat objects for comparison
    n_s1_predict = Nansat.from_domain(dst_dom, array = s1_dst_dom_S_hv)
    n_s2 = Nansat.from_domain(dst_dom, array = s2_dst_dom_hv)
    
    # 5.2. Create directory for saving plots 
    distort_plots_dir = os.path.join(output_dir_path, f"distort_error_plots")
    try:
        os.makedirs(distort_plots_dir, exist_ok=True)
        print(f"Successfully created {distort_plots_dir}")
    except Exception as e:
        print(f"Failed to create {distort_plots_dir}. Error: {e}")
    
    # Creating directory for saving distortion plots
    sar_distort_plots_dir = os.path.join(distort_plots_dir, f"sar_distort_plots")
    os.makedirs(sar_distort_plots_dir, exist_ok=True)
        
    # 5.3. Calculate realibility indexes 
    # 5.3.1. Run feature tracking and plot results 
    c1_alg_hv, r1_alg_hv, c2_alg_hv, r2_alg_hv = SAR1_SAR2_drift_retrieval.run_feature_tracking(n_s1_predict, n_s2, sar_distort_plots_dir, FT=False)
    
    # 5.3.2. Run pattern matching and plot results
    upm_alg_hv, vpm_alg_hv, apm_alg_hv, rpm_alg_hv, hpm_alg_hv, ssim_alg_hv, lon2pm_alg_hv, lat2pm_alg_hv = SAR1_SAR2_drift_retrieval.run_pattern_matching(sar_distort_plots_dir, x, y, 
                                                               lon1pm, lat1pm, n_s1_predict, c1_alg_hv, r1_alg_hv, n_s2, c2_alg_hv, r2_alg_hv, srs, 
                                                               #min_border=200,
                                                               #max_border=200,
                                                               #angles=[-50, -45, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 45, 50])
                                                               min_border=10, #test
                                                               max_border=10, #test
                                                               angles=[0]) #test
                                                               #angles=[-15,-12,-9,-6, -3, 0, 3, 6, 9, 12, 15]) #light
                                                               
    # Calculate gpi mask
    gpi1_alg_hv, gpi2_alg_hv = SAR1_SAR2_drift_retrieval.get_good_pixel_indices(hpm_alg_hv, h_threshold=hessian, neighbors_threshold=neighbors)
    
    # 5.4. Save comparison results, its parameters and filtering arrays to npy files
    save_name = 'sar_distort_error_data'
    sar_distort_data_error_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name,
                                                                             upm=upm_alg_hv, vpm=vpm_alg_hv, apm=apm_alg_hv, rpm=rpm_alg_hv, 
                                                                             hpm=hpm_alg_hv, ssim=ssim_alg_hv, lon2pm=lon2pm_alg_hv, 
                                                                             lat2pm=lat2pm_alg_hv, gpi1=gpi1_alg_hv, gpi2=gpi2_alg_hv)


    #print("4. Calculated quality parametrs (corr, hess, ssim) for the predicted SAR2 (by calculating pattern matching on SAR2 and SAR2_predicted.")
    #======================================================================
    # 6. Prepare model data for retrieving drift fields.
    #----------------------------------------------------------------------
    
    # 6.1.xtract time period based on teh SAR pair timestamp
    
    # SAR images timestamps
    t_sar1 = pair[0].timestamp
    t_sar2 = pair[1].timestamp
    
    # Rounding the SAR timestamps to align with the nearest whole hour of model timestamps
    t_start = model_data_processing.round_start_time(t_sar1)
    t_end = model_data_processing.round_end_time(t_sar2)

    print(f'SAR1 time is {t_sar1}, Model start time for the time period is {t_start}')
    print(f'SAR2 time is {t_sar2}, Model end time for the time period is {t_end}')
    
    # 6.2. Set the time period for extracting hourly model data
    time_period = pd.date_range(t_start, t_end, freq='H')

    # 6.3. Calculate the difference between model start and end time and SAR1 and SAR2 timestamps
    time_diff_start, time_diff_end, total_time_diff = model_data_processing.time_difference(t_sar1, t_sar2,  t_start, t_end)

    # 6.4. Calculate a hourly rolling average for 24 ensembles
    avg_ice_u, avg_ice_v =  model_data_processing.rolling_avg_24_ensembles(jt, time_period, min_row, max_row, min_col, max_col)
    
    # 6.5. Calculating cummulative (integrated) drift for the subset extent

    xx_b_subset, yy_b_subset, cum_dx_b_subset, cum_dy_b_subset = model_data_processing.cumulative_ice_displacement(X_subset, Y_subset, x, y, avg_ice_u, avg_ice_v, time_period, time_diff_start,time_diff_end)

    # 6.6. Get the final integrated displacement
    model_u = np.reshape(cum_dx_b_subset[-1], x.shape)
    model_v = np.reshape(cum_dy_b_subset[-1], x.shape)
    x2 = np.reshape(xx_b_subset[-1], x.shape)
    y2 = np.reshape(yy_b_subset[-1], x.shape)
    
    # 6.7. Save final drift, its parameters to npy files
    save_name = 'mod_drift_output'
    mod_drift_output_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name,
                                                                             model_u=model_u, model_v=model_v,
                                                                         y2=y2, x2=x2)
    # 6.8. Save the plot with model drift (using sar drift colourbar range)
    model_data_processing.plot_model_drift_results(drift_plot_save_path, x, y, model_u, model_v, disp_legend_min, disp_legend_max)
    model_data_processing.plot_model_drift_gpi_results(drift_plot_save_path, x, y, hpm, upm, vpm, gpi2, model_u, model_v, disp_legend_min, disp_legend_max)
    
    #print("5. Model data for retrieving drift fields prepared.")
    
    #======================================================================
    # 7. Warp SAR1 image with the model SAR drift and compare all SARs in the comparison domain
    #----------------------------------------------------------------------

    # 7.1. Warp
    # Warp SAR1 with model drift compenstaion/displacement
    # Create individual masks for non-NaN values in each array
    mask_u = np.isnan(model_u)
    mask_v = np.isnan(model_v)
    mask = mask_u & mask_v # mask out low quality or NaN
    
    s1_dst_dom_S_hv = warping_with_domain.warp_with_uv(n1_hv, n1_hv[1], mod_dom, model_u, model_v, mask, dst_dom) #warp_with_uv
    s1_dst_dom_S_hh = warping_with_domain.warp_with_uv(n1_hh, n1_hh[1], mod_dom, model_u, model_v, mask, dst_dom) #warp_with_uv
    

    # Warp SAR2 to the comparison domain
    s2_dst_dom_hv = warping_with_domain.warp(n2_hv, n2_hv[1], dst_dom)
    s2_dst_dom_hh = warping_with_domain.warp(n2_hh, n2_hh[1], dst_dom)

    # Warp SAR1 to the comparison domain for visualisation
    s1_dst_dom_hv = warping_with_domain.warp(n1_hv, n1_hv[1], dst_dom)
    s1_dst_dom_hh = warping_with_domain.warp(n1_hh, n1_hh[1], dst_dom)
    
    # 7.2. Plot warping results
    warping_with_domain.plot_sar_forecast_images(warping_plots_save_dir, 
                                                 "forecast_with_mod_ref_drift", 
                                                 s1_dst_dom_hv, s1_dst_dom_hh,
                                                 s2_dst_dom_hv,s2_dst_dom_hh,
                                                 s1_dst_dom_S_hv, s1_dst_dom_S_hh,
                                                 gamma_value=1.2)

    # 7.3. Save warped array to data output
    save_name = 'mod_warped_arrays'
    sar_distort_data_error_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name, 
                                                                                mod_warped_hv=s1_dst_dom_S_hv, mod_warped_hh=s1_dst_dom_S_hh)
    
    
    
    
    #======================================================================
    # 8. Calculate model warping quality parametrs (corr, hess, ssim) for the predicted SAR2 (by calculating pattern matching on SAR2 and SAR2_predicted)
    #----------------------------------------------------------------------
    
    # 8.1. Make new nansat objects for comparison
    n_s1_predict = Nansat.from_domain(dst_dom, array = s1_dst_dom_S_hv)
    n_s2 = Nansat.from_domain(dst_dom, array = s2_dst_dom_hv)

    # 8.2. Create directory for saving plots 
    mod_distort_plots_dir = os.path.join(distort_plots_dir, f"model_distortion_plots")
    try:
        os.makedirs(mod_distort_plots_dir, exist_ok=True)
        print(f"Successfully created {mod_distort_plots_dir}")
    except Exception as e:
        print(f"Failed to create {mod_distort_plots_dir}. Error: {e}")

    # Calculate realibility indexes 
    # 8.4. Run feature tracking on a regular grid for ares where drift errors wasn't strong but FT struggles
    c1_mod_hv, r1_mod_hv, c2_mod_hv, r2_mod_hv = SAR1_SAR2_drift_retrieval.run_feature_tracking(n_s1_predict, n_s2, mod_distort_plots_dir, FT=False)

    # 8.5. Run pattern matching and plot results
    upm_mod_hv, vpm_mod_hv, apm_mod_hv, rpm_mod_hv, hpm_mod_hv, ssim_mod_hv, lon2pm_mod_hv, lat2pm_mod_hv = SAR1_SAR2_drift_retrieval.run_pattern_matching(mod_distort_plots_dir, x, y, 
                                                               lon1pm, lat1pm, n_s1_predict, c1_mod_hv, r1_mod_hv, n_s2, c2_mod_hv, r2_mod_hv, srs, 
                                                               #min_border=200,
                                                               #max_border=200,
                                                               #angles=[-50, -45, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 45, 50])
                                                               min_border=10, #test
                                                               max_border=10, #test
                                                               angles=[0]) #test
                                                               
    
    # Calculate gpi mask
    gpi1_mod_hv, gpi2_mod_hv = SAR1_SAR2_drift_retrieval.get_good_pixel_indices(hpm_mod_hv, h_threshold=hessian, neighbors_threshold=neighbors)
        
    
    # 8.6. Save comparison results, its parameters and filtering arrays to npy files
    save_name = 'model_distort_error_data'
    mod_distort_data_error_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name,
                                                                             upm=upm_mod_hv, vpm=vpm_mod_hv, apm=apm_mod_hv, rpm=rpm_mod_hv, 
                                                                             hpm=hpm_mod_hv, ssim=ssim_mod_hv, lon2pm=lon2pm_mod_hv, 
                                                                             lat2pm=lat2pm_mod_hv, gpi1=gpi1_mod_hv, gpi2=gpi2_mod_hv)
    
    # Create folder for saving plots for PM with FT
    mod_distort_plots_dir = os.path.join(distort_plots_dir, f"model_distortion_error_ft_plots")
    try:
        os.makedirs(mod_distort_plots_dir, exist_ok=True)
        print(f"Successfully created {mod_distort_plots_dir}")
    except Exception as e:
        print(f"Failed to create {mod_distort_plots_dir}. Error: {e}")
    
    # 8.7. Run feature tracking with FT on for areas where patterns are good but ice drifted too far beyond search window border
    c1_mod_ft_hv, r1_mod_ft_hv, c2_mod_ft_hv, r2_mod_ft_hv = SAR1_SAR2_drift_retrieval.run_feature_tracking(n_s1_predict, n_s2, mod_distort_plots_dir, FT=True)

    # 8.8. Run pattern matching and plot results
    upm_mod_ft_hv, vpm_mod_ft_hv, apm_mod_ft_hv, rpm_mod_ft_hv, hpm_mod_ft_hv, ssim_mod_ft_hv, lon2pm_mod_ft_hv, lat2pm_mod_ft_hv = SAR1_SAR2_drift_retrieval.run_pattern_matching(mod_distort_plots_dir, x, y, 
                                                               lon1pm, lat1pm, n_s1_predict,  c1_mod_ft_hv, r1_mod_ft_hv, n_s2, c2_mod_ft_hv, r2_mod_ft_hv, srs, 
                                                               #min_border=200,
                                                               #max_border=200,
                                                               #angles=[-50, -45, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 45, 50])
                                                               min_border=10, #test
                                                               max_border=10, #test
                                                               angles=[0]) #test
                                                               
    # Calculate gpi mask
    gpi1_mod_ft_hv, gpi2_mod_ft_hv = SAR1_SAR2_drift_retrieval.get_good_pixel_indices(hpm_mod_ft_hv, h_threshold=hessian, neighbors_threshold=neighbors)
    
    
    # 8.9. Save comparison results, its parameters and filtering arrays to npy files
    save_name = 'model_distort_error_data_ft'
    mod_distort_data_error_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name,
                                                                             upm=upm_mod_ft_hv, vpm=vpm_mod_ft_hv, apm=apm_mod_ft_hv, rpm=rpm_mod_ft_hv, 
                                                                             hpm=hpm_mod_ft_hv, ssim=ssim_mod_ft_hv, lon2pm=lon2pm_mod_ft_hv, 
                                                                             lat2pm=lat2pm_mod_ft_hv, gpi1=gpi1_mod_ft_hv, gpi2=gpi2_mod_ft_hv)
    
    # 8.10. Combining results for FT off and on based on hessian threshold
    # Create folder for saving plots for combined method
    mod_distort_plots_dir = os.path.join(distort_plots_dir, f"model_distortion_error_combined_plots")
    try:
        os.makedirs(mod_distort_plots_dir, exist_ok=True)
        print(f"Successfully created {mod_distort_plots_dir}")
    except Exception as e:
        print(f"Failed to create {mod_distort_plots_dir}. Error: {e}")
        
    
    upm_mod_comb, vpm_mod_comb, apm_mod_comb, rpm_mod_comb, hpm_mod_comb, ssim_mod_comb, lon2pm_mod_comb, lat2pm_mod_comb = SAR1_SAR2_drift_retrieval.combine_based_on_hessian(output_dir_path,mod_distort_plots_dir, x, y, upm_mod_hv, vpm_mod_hv, apm_mod_hv, rpm_mod_hv, hpm_mod_hv, ssim_mod_hv, lon2pm_mod_hv, lat2pm_mod_hv, upm_mod_ft_hv, vpm_mod_ft_hv, apm_mod_ft_hv, rpm_mod_ft_hv, hpm_mod_ft_hv, ssim_mod_ft_hv, lon2pm_mod_ft_hv, lat2pm_mod_ft_hv)
   
    # Calculate gpi mask
    gpi1_mod_comb, gpi2_mod_comb = SAR1_SAR2_drift_retrieval.get_good_pixel_indices(hpm_mod_comb, h_threshold=hessian, neighbors_threshold=neighbors)
    
    # 8.11. Save comparison results, its parameters and filtering arrays to npy files
    save_name = 'model_distort_error_data_combined'
    mod_distort_data_error_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name,
                                                                             upm=upm_mod_comb, vpm=vpm_mod_comb, apm=apm_mod_comb, rpm=rpm_mod_comb, 
                                                                             hpm=hpm_mod_comb, ssim=ssim_mod_comb, lon2pm=lon2pm_mod_comb, 
                                                                             lat2pm=lat2pm_mod_comb, gpi1=gpi1_mod_comb, gpi2=gpi2_mod_comb)    
    
    
    #======================================================================
    # 9. Warping good drift mask using sar and model drift for further intercomparison. 
    #----------------------------------------------------------------------.
   
    good_pixels = gpi2
    mask_pm = ~good_pixels # mask out low quality or NaN
    
    # Convert the boolean mask to a float mask before warping
    float_gpi_mask = gpi2.astype(float)
    
    gpi_mask_sar_warp = warping_with_domain.warp_and_mask_with_uv(mod_dom, gpi2, mod_dom, upm, vpm, mask_pm, mod_dom, max_dist=0, fill_value=0)
    gpi_mask_mod_warp = warping_with_domain.warp_and_mask_with_uv(mod_dom, gpi2, mod_dom, model_u, model_v, mask_pm, mod_dom, max_dist=0, fill_value=0)
    
    save_name = 'warped_masks'
    sar_distort_data_error_path = SAR1_SAR2_drift_retrieval.save_results_to_npz(output_data_dir, save_name,
                                                                             sar_warped_mask=gpi_mask_sar_warp, model_warped_mask=gpi_mask_mod_warp)
    
    
    #======================================================================
    # 10. Comparing sar and model drift data.
    #----------------------------------------------------------------------
    
    # 10.1.  Replace inf with NaN before calculating the mean
    
    # Replace inf with nan in both arrays
    upm_no_inf = np.where(np.isinf(upm), np.nan, upm)
    vpm_no_inf = np.where(np.isinf(vpm), np.nan, vpm)
    
    
    # Plot histograms
    
    disp_model_b = np.sqrt(( model_u**2+ model_v**2)) 
    disp_alg = np.sqrt((upm_no_inf**2+ vpm_no_inf**2))

    disp_model_b = disp_model_b[gpi2].flatten()
    disp_alg = disp_alg[gpi2].flatten()

    plt.close('all')
    fig = plt.figure(figsize=(8,6))

    # Plotting Model displacements
    plt.hist(disp_model_b, bins=50, color='red', alpha=0.5, label= 'Barents Model Displacements')

    # Plotting Reference displacement
    plt.hist(disp_alg, bins=50, color='blue', alpha=0.5, label='SAR algorithm  Displacements')

    plt.xlabel('Displacement (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ice Drift Displacements (filtered)')
    plt.legend()
    
    fig.set_facecolor('white')
    
    # Save plot
    hist_save_path =  os.path.join(drift_plot_save_path, "drift_comparison_hist.png")
    
    fig.savefig(hist_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    


    # Print statistics for reference displacements
    percentile_95_ref = np.percentile(disp_alg, 95)
    mean_disp_ref = np.nanmean(disp_alg)
    min_disp_ref = np.nanmin(disp_alg)
    max_disp_ref = np.nanmax(disp_alg)

    print(f"Reference disp:")
    print(f"Mean disp: {mean_disp_ref/1000:.1f} km")
    print(f"Min disp: {min_disp_ref/1000:.1f} km")
    print(f"Max disp: {max_disp_ref/1000:.1f} km")
    print(f"95th percentile disp: {percentile_95_ref:.4f} km")

    # Print statistics for model displacements
    percentile_95_model = np.percentile(disp_model_b, 95)
    mean_disp_model = np.nanmean(disp_model_b)
    min_disp_model = np.nanmin(disp_model_b)
    max_disp_model = np.nanmax(disp_model_b)

    print("\nModel disp:")
    print(f"Mean disp: {mean_disp_model/1000:.1f} km")
    print(f"Min disp: {min_disp_model/1000:.1f} km")
    print(f"Max disp: {max_disp_model/1000:.1f} km")
    print(f"95th percentile disp: {percentile_95_model:.4f} km")
    
    
    #print("6. Histograms of SAR and model drift created.")
    
    
    #======================================================================
    # 10. Profiling
    #----------------------------------------------------------------------
    end_time = time.time()
    print(f"Pair {index} processed in {end_time - start_time:.2f} seconds.")
    
    pr.disable()  # Stop profiling
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()

    # Get the profiling results as a string and print it
    profiling_results = s.getvalue()
    # Save the stats to a file
    profiling_dir_path = os.path.join(output_dir_path, "profiling")
    os.makedirs(profiling_dir_path, exist_ok=True)
    save_path = os.path.join(profiling_dir_path, f"profile_results_pair{index}.prof")
    print(f'profiling path is {save_path}')
    ps.dump_stats(save_path)
    gc.collect()
    
