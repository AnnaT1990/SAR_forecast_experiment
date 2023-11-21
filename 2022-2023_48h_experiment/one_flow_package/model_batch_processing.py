import sys
sys.path.append("./modules")



# Import general modules
from nansat import Nansat, Domain, NSR
import os 
#for calculating time difference
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator

# Import temporal modules needed for testing plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%matplotlib inline

# Import SAR forecasting modules
import config
import s1_preparation
import domains_preparation
import SAR1_SAR2_drift_retrivial
import warping_with_domain
import model_data_proces

# Import variables
from config import path_to_HH_files, path_to_HV_files, safe_folder 
from config import output_folder, input_folder
from config import S1_prod_regex, S1_safe_regex
from config import lon, lat, X, Y, proj4, srs

# For cleaning up memory
import gc

import time

import warnings

# Ignore the runtime warnings related to 'Mean of empty slice' wen calculate rolling average
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')


#======================================================================
# 3
#----------------------------------------------------------------------
# 1. Prepare SAR pairs

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
# 7
#----------------------------------------------------------------------
# Define mod_dom and dst_dom outside the loop
mod_res = 2500
mod_dom = None  # Initialize to None
dst_res = 100
dst_dom = None  # Initialize to None


jt = model_data_proces.JohannesroThredds()

# Loop over all pairs and use enumerate to get an index for each pair
for index, pair in enumerate(sar_pairs, start=1):  # start=1 to have human-friendly indexing
    
    
    start_time = time.time()
    
    # 2.1. Prepare nansat objects and domains for HV polarisation
    n1_hv, n2_hv, output_dir_name, plots_dir_hv = domains_preparation.prepare_nansat_objects(
        pair[0], pair[1], output_folder, polarisation='HV')
    
    # Prepare nansat objects and domains for HH polarisation
    n1_hh, n2_hh, output_dir_name, plots_dir_hh = domains_preparation.prepare_nansat_objects(
        pair[0], pair[1], output_folder, polarisation='HH')
    

    # Additional processing steps
    # 2.2  Define model domain (mod_dom) for comparing drift and comparison (dst_dom) domain to compare SAR images (real and forecasted)
    
    '''
    # Pick this if want create mod and comparison domain for each pair (then they can differ)
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
    
    # Check if mod_dom and dst_dom are None, if so, define them based on the first pair
    if mod_dom is None:
        X_subset, Y_subset, lon_subset, lat_subset, min_row, max_row, min_col, max_col = domains_preparation.prepare_grid(n1_hv, n2_hv, srs, X, Y, lon, lat, buffer=0)
        mod_dom = Domain(srs, f'-te {min(X_subset.data)} {min(Y_subset.data) - mod_res * 2} {max(X_subset.data) + mod_res} {max(Y_subset.data)} -tr {mod_res} {mod_res}')
        lon1pm, lat1pm = mod_dom.get_geolocation_grids()
        x, y = mod_dom.get_geolocation_grids(dst_srs=srs)
    
    if dst_dom is None:
        dst_dom = Domain(srs, f'-te {min(X_subset.data)} {min(Y_subset.data) - dst_res * 2} {max(X_subset.data) + dst_res} {max(Y_subset.data)} -tr {dst_res} {dst_res}')
        
    
    domains_preparation.plot_borders(mod_dom, n1_hv, n2_hv, output_dir_name) # borders for hh and hv are the same
    # Checking that domains have the same borders
    
    rows1, cols1 = dst_dom.shape()
    print("dst_dom corner coordinates:", dst_dom.transform_points([0,cols1-1,0,cols1-1], [0,0,rows1-1,rows1-1], dst_srs=srs))
    
    rows1, cols1 = mod_dom.shape()
    print("mod_dom corner coordinates:", mod_dom.transform_points([0,cols1-1,0,cols1-1], [0,0,rows1-1,rows1-1], dst_srs=srs))
    
    
    # 3. Retrieve model drift data for the subset

    # 3.1. Extract teh time period based on teh SAR pair timestamp

    # SAR images timestamps
    t_sar1 = pair[0].timestamp
    t_sar2 = pair[1].timestamp

    # Rounding the SAR timestamps to align with the nearest whole hour of model timestamps
    t_start = model_data_processing.round_start_time(t_sar1)
    t_end = model_data_processing.round_end_time(t_sar2)

    print(f'SAR1 time is {t_sar1}, Model start time for the time period is {t_start}')
    print(f'SAR2 time is {t_sar2}, Model end time for the time period is {t_end}')

    # 3.2. Set the time period for extracting hourly model data

    time_period = pd.date_range(t_start, t_end, freq='H')

    # 3.3. Calculate the difference between model start and end time and SAR1 and SAR2 timestamps
    time_diff_start, time_diff_end, total_time_diff = model_data_processing.time_difference(t_sar1, t_sar2,  t_start, t_end)

    # 3.4. Calculate a rolling average

    avg_ice_u, avg_ice_v =  model_data_processing.rolling_avg_24_ensembles(jt, time_period, min_row, max_row, min_col, max_col)
    
    # 3.5. Calculating cummulative (integrated) drift for the subset extent

    xx_b_subset, yy_b_subset, cum_dx_b_subset, cum_dy_b_subset = model_data_processing.cumulative_ice_displacement(X_subset, Y_subset, x, y, avg_ice_u, avg_ice_v, time_period, time_diff_start,time_diff_end)

    # 3.6. Get the final integrated displacement
    model_u = np.reshape(cum_dx_b_subset[-1], x.shape)
    model_v = np.reshape(cum_dy_b_subset[-1], x.shape)
    x2 = np.reshape(xx_b_subset[-1], x.shape)
    y2 = np.reshape(yy_b_subset[-1], x.shape)
    
    # 3.7. Save final drift, its parameters to npy files
    save_name = 'mod_drift_output'
    sar_drift_output_path = SAR1_SAR2_drift_retrivial.save_sar_drift_results(output_dir_name, save_name,
                                                                             model_u=model_u, model_v=model_v,
                                                                         y2=y2, x2=x2)

    # 4. Warp SAR1 image with the reference sar drift and compare all arrays in the comparison distination domain

    # 4.1. Warp
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
    
    # 4.2. Plot warping results

    # Create General plots folder if it doesn't exist already (but it should)
    general_save_path = os.path.join(output_dir_name, "General_plots")
    os.makedirs(general_save_path, exist_ok=True)


    warping_with_domain.plot_sar_forecast_images(general_save_path, 
                                                 "Forecast_with_mod_ref_drift", 
                                             s1_dst_dom_hv, s2_dst_dom_hv, s1_dst_dom_S_hv,
                                             s1_dst_dom_hh, s2_dst_dom_hh, s1_dst_dom_S_hh,
                                             gamma_value=1.2)
    
    
    
    
    # 5. Calculate quality parametrs (corr, hess, ssim) for the predicted SAR2 (by calculating pattern matchin on SAR2 and SAR2_predicted)

    # 5.1. Make new nansat objects for comparison

    n_s1_predict = Nansat.from_domain(dst_dom, array = s1_dst_dom_S_hv)
    n_s2 = Nansat.from_domain(dst_dom, array = s2_dst_dom_hv)

    # 5.2. Create directory for saving plots 
    comparison_dir = os.path.join(output_dir_name, f"Model_distortion_error")
    try:
        os.makedirs(comparison_dir, exist_ok=True)
        print(f"Successfully created {comparison_dir}")
    except Exception as e:
        print(f"Failed to create {comparison_dir}. Error: {e}")

    # Calculate realibility indexes 

    # 5.4. Run feature tracking and plot results 
    c1_alg_hv, r1_alg_hv, c2_alg_hv, r2_alg_hv = SAR1_SAR2_drift_retrivial.run_feature_tracking(n_s1_predict, n_s2, comparison_dir)

    # 5.5. Run pattern matching and plot results
    upm_alg_hv, vpm_alg_hv, apm_alg_hv, rpm_alg_hv, hpm_alg_hv, ssim_alg_hv, lon2pm_alg_hv, lat2pm_alg_hv = SAR1_SAR2_drift_retrivial.run_pattern_matching(comparison_dir, x, y, 
                                                               lon1pm, lat1pm, n_s1_predict, c1_alg_hv, r1_alg_hv, n_s2, c2_alg_hv, r2_alg_hv, srs, 
                                                               min_border=200,
                                                               max_border=200,
                                                               #min_border=10, #test
                                                               #max_border=10, #test
                                                               #angles=[0]) #test
                                                               angles=[-50, -45, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 45, 50])



    
    end_time = time.time()
    print(f"Pair {index} processed in {end_time - start_time:.2f} seconds.")