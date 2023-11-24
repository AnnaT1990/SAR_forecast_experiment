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
import SAR1_SAR2_drift_retrivial
import warping_with_domain
import model_data_processing

# Import variables
from config import path_to_HH_files, path_to_HV_files, safe_folder 
from config import output_folder, input_folder
from config import S1_prod_regex, S1_safe_regex
from config import lon, lat, X, Y, proj4, srs
from config import hessian, neighbors

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
    
    # 2.1. Prepare nansat objects and domains for HV polarisation
    n1_hv, n2_hv, output_dir_name, plots_dir_hv = domains_preparation.prepare_nansat_objects(
        pair[0], pair[1], output_folder, polarisation='HV')
    
    # Prepare nansat objects and domains for HH polarisation
    n1_hh, n2_hh, output_dir_name, plots_dir_hh = domains_preparation.prepare_nansat_objects(
        pair[0], pair[1], output_folder, polarisation='HH')
    

    # 2.2  Define model domain (mod_dom) for comparing drift and comparison (dst_dom) domain to compare SAR images (real and forecasted)
 
    
    # Check if mod_dom and dst_dom are None, if so, define them based on the first pair
    if mod_dom is None:
        X_subset, Y_subset, lon_subset, lat_subset, min_row, max_row, min_col, max_col = domains_preparation.prepare_grid(n1_hv, n2_hv, srs, X, Y, lon, lat, buffer=0)
        mod_dom = Domain(srs, f'-te {min(X_subset.data)} {min(Y_subset.data) - mod_res * 2} {max(X_subset.data) + mod_res} {max(Y_subset.data)} -tr {mod_res} {mod_res}')
        lon1pm, lat1pm = mod_dom.get_geolocation_grids()
        x, y = mod_dom.get_geolocation_grids(dst_srs=srs)
    
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
    '''    
    
    drift_plot_save_path = domains_preparation.plot_borders(mod_dom, n1_hv, n2_hv, output_dir_name) # borders for hh and hv are the same
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
    c1_hv, r1_hv, c2_hv, r2_hv = SAR1_SAR2_drift_retrivial.run_feature_tracking(n1_hv, n2_hv, plots_dir_hv)
    
    #Run pattern matching and plot results
    upm_hv, vpm_hv, apm_hv, rpm_hv, hpm_hv, ssim_hv, lon2pm_hv, lat2pm_hv = SAR1_SAR2_drift_retrivial.run_pattern_matching(plots_dir_hv, x, y, 
                                                               lon1pm, lat1pm, n1_hv, c1_hv, r1_hv, n2_hv, c2_hv, r2_hv, srs, 
                                                               min_border=200,
                                                               max_border=200,
                                                               #min_border=10, #test
                                                               #max_border=10, #test
                                                               #angles=[0]) #test
                                                               angles=[-50, -45, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 45, 50])
    # 3.2. Run feature tracking and pattern matching for HH
    
    # HH Processing
    # Run feature tracking and plot results 
    c1_hh, r1_hh, c2_hh, r2_hh = SAR1_SAR2_drift_retrivial.run_feature_tracking(n1_hh, n2_hh, plots_dir_hh)
    
    #Run pattern matching and plot results
    upm_hh, vpm_hh, apm_hh, rpm_hh, hpm_hh, ssim_hh, lon2pm_hh, lat2pm_hh = SAR1_SAR2_drift_retrivial.run_pattern_matching(plots_dir_hh, x, y, 
                                                               lon1pm, lat1pm, n1_hh, c1_hh, r1_hh, n2_hh, c2_hh, r2_hh,srs, 
                                                               min_border=200,
                                                               max_border=200,
                                                               #min_border=10, #test
                                                               #max_border=10, #test
                                                               #angles=[0]) #test
                                                               angles=[-50, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 50 ])
    
    
    # 3.3. Get combined drift and all textural parameters
    
    # Combining hh and hv results based on hessian threshold
    upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm = SAR1_SAR2_drift_retrivial.combine_hh_hv(output_dir_name, x, y, upm_hh, vpm_hh, apm_hh, rpm_hh, hpm_hh, ssim_hh, lon2pm_hh, lat2pm_hh,
                                  upm_hv, vpm_hv, apm_hv, rpm_hv, hpm_hv, ssim_hv, lon2pm_hv, lat2pm_hv)
   
    # 3.4.  Filter drift data with the good pixel indices based on hessian and neighbor thresholds.
    
    #Returns:
    #    - gpi1: Good pixel index based on hessian value
    #    - gpi2: Good pixel index combining hessian and neighbors count 

    
    gpi1, gpi2 = SAR1_SAR2_drift_retrivial.get_good_pixel_indices(hpm, h_threshold=hessian, neighbors_threshold=neighbors)
    
        
    # Plot the filtering results
    drift_plot_save_path, sar_disp_min, sar_disp_max = SAR1_SAR2_drift_retrivial.plot_filter_results(output_dir_name, x, y, hpm, upm, vpm, gpi1, gpi2, hessian, neighbors)
    
    
    #  Save final reference drift, its parameters and filtering arrays to npy files
    save_name = 'sar_drift_output'
    sar_drift_output_path = SAR1_SAR2_drift_retrivial.save_sar_drift_results(output_dir_name, save_name,
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
    
    # Warp SAR2 to the comparison domain
    s2_dst_dom_hv = warping_with_domain.warp(n2_hv, n2_hv[1], dst_dom)
    s2_dst_dom_hh = warping_with_domain.warp(n2_hh, n2_hh[1], dst_dom)
    
    # Warp SAR1 to the comparison domain for visualisation
    s1_dst_dom_hv = warping_with_domain.warp(n1_hv, n1_hv[1], dst_dom)
    s1_dst_dom_hh = warping_with_domain.warp(n1_hh, n1_hh[1], dst_dom)
    

    # 4.2. Plot warping results
    warping_plots_save_dir = os.path.join(output_dir_name, "warping_plots")
    os.makedirs(warping_plots_save_dir, exist_ok=True)
    warping_with_domain.plot_sar_forecast_images(warping_plots_save_dir, 
                                                 "Forecast_with_sar_ref_drift", 
                                                 s1_dst_dom_hv, s2_dst_dom_hv, s1_dst_dom_S_hv,
                                                 s1_dst_dom_hh, s2_dst_dom_hh, s1_dst_dom_S_hh,
                                                 gamma_value=1.2)
    
    #print("3. Warped SAR1 image with the reference SAR drift.")
        
    #======================================================================
    # 5. Calculate  sar warping quality parametrs (corr, hess, ssim) for the predicted SAR2 (by calculating pattern matching on SAR2 and SAR2_predicted)
    #----------------------------------------------------------------------
    
    # 5.1. Make new nansat objects for comparison
    n_s1_predict = Nansat.from_domain(dst_dom, array = s1_dst_dom_S_hv)
    n_s2 = Nansat.from_domain(dst_dom, array = s2_dst_dom_hv)
    
    # 5.2. Create directory for saving plots 
    sar_distort_plots_dir = os.path.join(output_dir_name, f"sar_distort_error_plots")
    try:
        os.makedirs(sar_distort_plots_dir, exist_ok=True)
        print(f"Successfully created {sar_distort_plots_dir}")
    except Exception as e:
        print(f"Failed to create {sar_distort_plots_dir}. Error: {e}")
        
    # 5.3. Calculate realibility indexes 
    # 5.3.1. Run feature tracking and plot results 
    c1_alg_hv, r1_alg_hv, c2_alg_hv, r2_alg_hv = SAR1_SAR2_drift_retrivial.run_feature_tracking(n_s1_predict, n_s2, sar_distort_plots_dir)
    
    # 5.3.2. Run pattern matching and plot results
    upm_alg_hv, vpm_alg_hv, apm_alg_hv, rpm_alg_hv, hpm_alg_hv, ssim_alg_hv, lon2pm_alg_hv, lat2pm_alg_hv = SAR1_SAR2_drift_retrivial.run_pattern_matching(sar_distort_plots_dir, x, y, 
                                                               lon1pm, lat1pm, n_s1_predict, c1_alg_hv, r1_alg_hv, n_s2, c2_alg_hv, r2_alg_hv, srs, 
                                                               min_border=200,
                                                               max_border=200,
                                                               #min_border=10, #test
                                                               #max_border=10, #test
                                                               #angles=[0]) #test
                                                               #angles=[-15,-12,-9,-6, -3, 0, 3, 6, 9, 12, 15]) #light
                                                               angles=[-50, -45, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 45, 50])
    
    # 5.4. Save comparison results, its parameters and filtering arrays to npy files
    save_name = 'sar_distort_error_data'
    sar_distort_data_error_path = SAR1_SAR2_drift_retrivial.save_sar_drift_results(output_dir_name, save_name,
                                                                             upm=upm_alg_hv, vpm=vpm_alg_hv, apm=apm_alg_hv, rpm=rpm_alg_hv, 
                                                                             hpm=hpm_alg_hv, ssim=ssim_alg_hv, lon2pm=lon2pm_alg_hv, 
                                                                             lat2pm=lat2pm_alg_hv, gpi1=gpi1, gpi2=gpi2)


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
    mod_drift_output_path = SAR1_SAR2_drift_retrivial.save_sar_drift_results(output_dir_name, save_name,
                                                                             model_u=model_u, model_v=model_v,
                                                                         y2=y2, x2=x2)
    # 6.8. Save the plot with model drift (using sar drift colourbar range)
    model_data_processing.plot_model_drift_results(drift_plot_save_path, x, y, model_u, model_v, sar_disp_min, sar_disp_max)
    model_data_processing.plot_model_drift_gpi_results(drift_plot_save_path, x, y, hpm, upm, vpm, gpi2, model_u, model_v, sar_disp_min, sar_disp_max)
    
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
    s1_dst_dom_S_hv = warping_with_domain.warp_with_uv(n1_hv, n1_hv[1], mod_dom, model_u, model_v, mask, dst_dom)
    s1_dst_dom_S_hh = warping_with_domain.warp_with_uv(n1_hh, n1_hh[1], mod_dom, model_u, model_v, mask, dst_dom)

    # Warp SAR2 to the comparison domain
    s2_dst_dom_hv = warping_with_domain.warp(n2_hv, n2_hv[1], dst_dom)
    s2_dst_dom_hh = warping_with_domain.warp(n2_hh, n2_hh[1], dst_dom)

    # Warp SAR1 to the comparison domain for visualisation
    s1_dst_dom_hv = warping_with_domain.warp(n1_hv, n1_hv[1], dst_dom)
    s1_dst_dom_hh = warping_with_domain.warp(n1_hh, n1_hh[1], dst_dom)
    
    # 7.2. Plot warping results


    warping_with_domain.plot_sar_forecast_images(warping_plots_save_dir, 
                                                 "Forecast_with_mod_ref_drift", 
                                             s1_dst_dom_hv, s2_dst_dom_hv, s1_dst_dom_S_hv,
                                             s1_dst_dom_hh, s2_dst_dom_hh, s1_dst_dom_S_hh,
                                             gamma_value=1.2)
    #======================================================================
    # 8. Calculate model warping quality parametrs (corr, hess, ssim) for the predicted SAR2 (by calculating pattern matching on SAR2 and SAR2_predicted)
    #----------------------------------------------------------------------
    
    # 8.1. Make new nansat objects for comparison
    n_s1_predict = Nansat.from_domain(dst_dom, array = s1_dst_dom_S_hv)
    n_s2 = Nansat.from_domain(dst_dom, array = s2_dst_dom_hv)

    # 8.2. Create directory for saving plots 
    mod_distort_plots_dir = os.path.join(output_dir_name, f"model_distortion_error_plots")
    try:
        os.makedirs(mod_distort_plots_dir, exist_ok=True)
        print(f"Successfully created {mod_distort_plots_dir}")
    except Exception as e:
        print(f"Failed to create {mod_distort_plots_dir}. Error: {e}")

    # Calculate realibility indexes 
    # 8.4. Run feature tracking and plot results 
    c1_alg_hv, r1_alg_hv, c2_alg_hv, r2_alg_hv = SAR1_SAR2_drift_retrivial.run_feature_tracking(n_s1_predict, n_s2, mod_distort_plots_dir)

    # 8.5. Run pattern matching and plot results
    upm_alg_hv, vpm_alg_hv, apm_alg_hv, rpm_alg_hv, hpm_alg_hv, ssim_alg_hv, lon2pm_alg_hv, lat2pm_alg_hv = SAR1_SAR2_drift_retrivial.run_pattern_matching(mod_distort_plots_dir, x, y, 
                                                               lon1pm, lat1pm, n_s1_predict, c1_alg_hv, r1_alg_hv, n_s2, c2_alg_hv, r2_alg_hv, srs, 
                                                               min_border=200,
                                                               max_border=200,
                                                               #min_border=10, #test
                                                               #max_border=10, #test
                                                               #angles=[0]) #test
                                                               angles=[-50, -45, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 45, 50])
    
    # 8.6. Save comparison results, its parameters and filtering arrays to npy files
    save_name = 'model_distort_error_data'
    mod_distort_data_error_path = SAR1_SAR2_drift_retrivial.save_sar_drift_results(output_dir_name, save_name,
                                                                             upm=upm_alg_hv, vpm=vpm_alg_hv, apm=apm_alg_hv, rpm=rpm_alg_hv, 
                                                                             hpm=hpm_alg_hv, ssim=ssim_alg_hv, lon2pm=lon2pm_alg_hv, 
                                                                             lat2pm=lat2pm_alg_hv, gpi1=gpi1, gpi2=gpi2)
    
    #======================================================================
    # 9. Comparing sar and model drift data.
    #----------------------------------------------------------------------
    
    # 9.1.  Replace inf with NaN before calculating the mean
    
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
    profiling_dir_path = os.path.join(output_dir_name, "profiling")
    os.makedirs(profiling_dir_path, exist_ok=True)
    save_path = os.path.join(profiling_dir_path, f"profile_results_pair{index}.prof")
    print(f'profiling path is {save_path}')
    ps.dump_stats(save_path)
    gc.collect()
    
