#======================================================================
# 1
#----------------------------------------------------------------------
import sys
sys.path.append("./modules")


#======================================================================
# 2
#----------------------------------------------------------------------
# Import general modules
from nansat import Nansat, Domain, NSR
import os 

# Import temporal modules needed for testing plotting
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

# Import SAR forecasting modules
import config
import s1_preparation
import domains_preparation
import SAR1_SAR2_drift_retrivial
import warping_with_domain

# Import variables
from config import path_to_HH_files, path_to_HV_files, safe_folder 
from config import output_folder, input_folder
from config import S1_prod_regex, S1_safe_regex
from config import lon, lat, X, Y, proj4, srs

# For cleaning up memory
import gc

import time


import cProfile
import pstats
from io import StringIO

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



# Loop over all pairs and use enumerate to get an index for each pair
for index, pair in enumerate(sar_pairs, start=1):  # start=1 to have human-friendly indexing
    
    # Create a Profile object
    pr = cProfile.Profile()
    pr.enable()  # Start profiling
    
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
    
    # 3.   Retrieve reference drift
    # 3.1. Run feature tracking and pattern matching for HV
    
   
    # Run feature tracking and plot results 
    c1_hv, r1_hv, c2_hv, r2_hv = SAR1_SAR2_drift_retrivial.run_feature_tracking(n1_hv, n2_hv, plots_dir_hv)
    
    #Run pattern matching and plot results
    upm_hv, vpm_hv, apm_hv, rpm_hv, hpm_hv, ssim_hv, lon2pm_hv, lat2pm_hv = SAR1_SAR2_drift_retrivial.run_pattern_matching(plots_dir_hv, x, y, 
                                                               lon1pm, lat1pm, n1_hv, c1_hv, r1_hv, n2_hv, c2_hv, r2_hv, srs, 
                                                               min_border=200,
                                                               max_border=200,
                                                               #min_border=6, #test
                                                               #max_border=6, #test
                                                               #angles=[-6, -3, 0, 3, 6]) #test
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
                                                               #min_border=6, #test
                                                               #max_border=6, #test
                                                               #angles=[-6, -3, 0, 3, 6]) #test
                                                               angles=[-50, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 50 ])
    
    
    # 3.3. Get combined drift and all textural parameters
    
    # Combining hh and hv results based on hessian threshold
    upm, vpm, apm, rpm, hpm, ssim, lon2pm, lat2pm = SAR1_SAR2_drift_retrivial.combine_hh_hv(output_dir_name, x, y, upm_hh, vpm_hh, apm_hh, rpm_hh, hpm_hh, ssim_hh, lon2pm_hh, lat2pm_hh,
                                  upm_hv, vpm_hv, apm_hv, rpm_hv, hpm_hv, ssim_hv, lon2pm_hv, lat2pm_hv)
    # 3.4.  Get good pixel indices based on hessian and neighbor thresholds.
    
    #Returns:
    #    - gpi1: Good pixel index based on hessian value
    #    - gpi2: Good pixel index combining hessian and neighbors count 
    
    hessian=8
    neighbors=2
    
    gpi1, gpi2 = SAR1_SAR2_drift_retrivial.get_good_pixel_indices(hpm, h_threshold=hessian, neighbors_threshold=neighbors)
    
        
    # Plot the filtering results
    general_plots_path = SAR1_SAR2_drift_retrivial.plot_filter_results(output_dir_name, x, y, hpm, upm, vpm, gpi1, gpi2, hessian, neighbors)
    
    
    #  Save final drift, its parameters and filtering arrays to npy files
    save_name = 'sar_ref_drift_output'
    sar_drift_output_path = SAR1_SAR2_drift_retrivial.save_sar_drift_results(output_dir_name, save_name,
                                                                             upm=upm, vpm=vpm, apm=apm, rpm=rpm, 
                                                                             hpm=hpm, ssim=ssim, lon2pm=lon2pm, 
                                                                             lat2pm=lat2pm, gpi1=gpi1, gpi2=gpi2)
    # 4. Warp SAR1 image with the reference sar drift and compare all arrays in the comparison distination domain
    
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
    warping_with_domain.plot_sar_forecast_images(general_plots_path, 
                                                 "Forecast_with_sar_ref_drift", 
                                                 s1_dst_dom_hv, s2_dst_dom_hv, s1_dst_dom_S_hv,
                                                 s1_dst_dom_hh, s2_dst_dom_hh, s1_dst_dom_S_hh,
                                                 gamma_value=1.2)
    # 5. Calculate quality parametrs (corr, hess, ssim) for the predicted SAR2 (by calculating pattern matchin on SAR2 and SAR2_predicted)
    
    # 5.1. Make new nansat objects for comparison
    
    n_s1_predict = Nansat.from_domain(dst_dom, array = s1_dst_dom_S_hv)
    n_s2 = Nansat.from_domain(dst_dom, array = s2_dst_dom_hv)
    
    # 5.2. Create directory for saving plots 
    comparison_dir = os.path.join(output_dir_name, f"comparison_plots")
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
                                                               #min_border=6, #test
                                                               #max_border=6, #test
                                                               #angles=[-6, -3, 0, 3, 6]) #test
                                                               angles=[-15,-12,-9,-6, -3, 0, 3, 6, 9, 12, 15]) #light
                                                               #angles=[-50, -45, -40, -35, -30, -25, -20, -15,-12, -9,-6, -3, 0, 3, 6, 9, 12,15, 20, 25, 30, 35, 40, 45, 50])
    
    # 5.6. Save comparison results, its parameters and filtering arrays to npy files
    save_name = 'sar_drift_forecast_quality'
    sar_drift_output_path = SAR1_SAR2_drift_retrivial.save_sar_drift_results(output_dir_name, save_name,
                                                                             upm=upm_alg_hv, vpm=vpm_alg_hv, apm=apm_alg_hv, rpm=rpm_alg_hv, 
                                                                             hpm=hpm_alg_hv, ssim=ssim_alg_hv, lon2pm=lon2pm_alg_hv, 
                                                                             lat2pm=lat2pm_alg_hv, gpi1=gpi1, gpi2=gpi2)

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
    

