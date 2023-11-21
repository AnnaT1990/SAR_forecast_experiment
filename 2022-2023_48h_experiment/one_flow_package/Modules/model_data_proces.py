# Module

# ---------------------------------------------------------------------- #
# Name :        model_data_processing.py
# Purpose :     Prepare model data for the forecast and comparison with reference data
# ---------------------------------------------------------------------- #

import re
import datetime
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator

from collections import defaultdict
from datetime import datetime
import requests
import xarray as xr
from bs4 import BeautifulSoup

from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator


''' 
Comments
----------

The JohannesroThredds class provides a convenient interface to interact with the THREDDS data server used by MET Norway. 
It is tailored to work with the ensemble Prediction System (EPS) datasets for the Barents Sea region.
The class facilitates fetching dataset URLs, sorting them by date, and extracting specific variable data from the datasets for given spatial and temporal dimensions.

   Attributes
    ----------
    top_url : str
        - The top-level URL of the THREDDS catalog.
    required_part : str
        - A partial URL string that is required to be in the dataset links to filter relevant datasets.
    opendap_base_url : str
        - The base URL for the OPeNDAP data access protocol used to retrieve dataset contents.
    dates : numpy.ndarray
        - An array of numpy datetime64 objects representing the sorted dates of available datasets.
    urls : list
        - A list of complete URLs to the datasets sorted by date.

    Methods
    -------
    get_last_dates_urls(timestamp, n=4) : tuple
        - Returns the last 'n' dates and corresponding dataset URLs closest to a specified timestamp.
    get_var_data(timestamp, row_min=0, row_max=None, col_min=0, col_max=None, var_names=('ice_u', 'ice_v')) : dict
        - Retrieves and returns a dictionary with the data for specified variables from datasets closest to the given timestamp, with optional spatial subsetting.
   
'''
    

class JohannesroThredds:
    # Static class attributes defining the base URLs for catalog access and data retrieval.
    top_url = 'https://thredds.met.no/thredds/catalog/metusers/johannesro/eps/catalog.html'
    required_part = 'catalog.html?dataset=metusers/johannesro/eps/barents_eps'
    opendap_base_url = 'https://thredds.met.no/thredds/dodsC/metusers/johannesro/eps/'

    def __init__(self, top_url=None):
        if top_url:
            self.top_url = top_url
        with requests.get(self.top_url, allow_redirects=True) as r:
            soup = BeautifulSoup(r.content, 'html.parser')    
        alla = soup.find_all('a')        
        hrefs = [a.get('href') for a in alla if self.required_part in a.get('href')]
        dates = [datetime.strptime(href.rstrip('Z.nc').split('_')[-1], '%Y%m%dT%H') for href in hrefs]
        dates = np.array([np.datetime64(date) for date in dates])
        sort_idx = np.argsort(dates)
        self.dates = dates[sort_idx]
        self.urls = [self.opendap_base_url + hrefs[i].split('/')[-1] for i in sort_idx]

    def get_last_dates_urls(self, timestamp, n=4):
        """
        Fetches the last 'n' dates and corresponding dataset URLs based on proximity
        to a specified timestamp.
        
        Parameters:
        timestamp: The reference timestamp to compare against the dataset timestamps.
        n: The number of dataset instances to return.
        
        Returns:
        A tuple containing a list of dates and a list of dataset URLs.
        """
        i = np.argmin(np.abs(self.dates - np.datetime64(timestamp)))
        ii = range(i-n+1, i+1)
        return [self.dates[i] for i in ii], [self.urls[i] for i in ii]

    
    
    def get_var_data(self, timestamp, row_min=0, row_max=None, col_min=0, col_max=None, var_names=('ice_u', 'ice_v')):
        """
        Retrieves the data for specified variables from the datasets that are closest
        to a given timestamp. Spatial subsetting is also supported.
        
        Parameters:
        timestamp: The reference timestamp for data retrieval.
        row_min: The minimum row index for spatial subsetting (inclusive).
        row_max: The maximum row index for spatial subsetting (exclusive).
        col_min: The minimum column index for spatial subsetting (inclusive).
        col_max: The maximum column index for spatial subsetting (exclusive).
        var_names: A tuple of variable names to retrieve data for.
        
         Returns:
         A dictionary where keys are variable names and values are numpy arrays of retrieved data.
         """
        dates, urls = self.get_last_dates_urls(timestamp)
        var_data = defaultdict(list)
        for url in urls:
            #print('Load from ', url)
            with xr.open_dataset(url) as ds:
                time = ds['time'].to_numpy()
                time_idx = np.argmin(np.abs(time - np.datetime64(timestamp)))
                #print('Time index: ', time_idx)
                for var_name in var_names:
                    var_data[var_name].append(
                        ds[var_name][time_idx, :, row_min:row_max+1, col_min:col_max+1].to_numpy()
                    )
        return {var_name: np.vstack(var_data[var_name]) for var_name in var_data}

    
###########################################################################
# Other functions
###########################################################################


def round_start_time(t_dt):
    """
    Rounds down the given timestamp to the nearest hour.

    Parameters:
    - t (str): The input timestamp in the format '%Y-%m-%dT%H:%M:%S'.

    Returns:
    - str: The rounded timestamp.
    """
    rounded_dt = t_dt.replace(minute=0, second=0)
    return rounded_dt

def round_end_time(t_dt):
    """
    Rounds up the given timestamp to the nearest hour.

    Parameters:
    - t (str): The input timestamp in the format '%Y-%m-%dT%H:%M:%S'.

    Returns:
    - str: The rounded timestamp.
    """
    rounded_dt = t_dt + timedelta(hours=1)
    rounded_dt = rounded_dt.replace(minute=0, second=0)
    return rounded_dt


def time_difference(t_sar1, t_sar2, t_start, t_end):
    import numpy as np
    """
    Calculates the time differences between SAR1 and SAR2 images timestamps and the model timestamps 
    at the beginning and end of the time period. This will be used to calculate the fraction 
    of the hourly model displacement in the first and last partial hour of the forecast.

    Parameters:
    - t_sar1 (str): The timestamp for SAR1.
    - t_sar2 (str): The timestamp for SAR2.
    - t_start (str): The timestamp for the beginning of the model hourly time period.
    - t_end (str):  The timestamp for the end of the model hourly time perio

    Returns: 
    - time_diff_start (int): Time difference between SAR1 and the start of the model time period in seconds.
    - time_diff_end (int): Time difference between SAR2 and the end of the model time period in seconds.
    - total_time_diff (int): Total time difference between SAR1 and SAR2 images in seconds.
    """
    
    time_diff_start = t_sar1 - t_start
    time_diff_end = t_end - t_sar2
    total_time_diff = t_sar2 - t_sar1
    
    time_diff_start = time_diff_start.seconds
    time_diff_end = time_diff_end.seconds
    total_time_diff = total_time_diff.seconds

    print(f'Time difference between SAR1 and the start of the model time period is {time_diff_start} seconds ({np.around(time_diff_start/60, 2)} minutes).') 
    print(f'Time difference between SAR2 and the end of the model time period is {time_diff_end} seconds ({np.around(time_diff_end/60, 2)} minutes).')
    print(f'Total time difference between SAR1 and SAR2 images is {total_time_diff} seconds ({np.around(total_time_diff/3600, 2)} hours).') 
    
    return time_diff_start, time_diff_end, total_time_diff

def rolling_avg_24_ensembles(jt, time_period, min_row, max_row, min_col, max_col):
    """
    Compute hourly ensemble averages of ice drift velocities over a given time range.

    Using a rolling averaging approach, this function calculates the average ice drift velocities 
    (specifically ice_u and ice_v) for each hour within a specified forecast period. For each 
    hourly timestamp, data is sourced from the four most recent ensemble forecast datasets. 
    This ensures the inclusion of the most up-to-date predictions. The function produces a 
    continuous time series of average ice drift velocities by always considering the latest 
    ensemble forecasts.

    Parameters:
    - start_time (str): Start of the desired time range in 'YYYY-MM-DD HH:MM:SS' format.
    - end_time (str): End of the desired time range in 'YYYY-MM-DD HH:MM:SS' format.
    - min_row, max_row, min_col, max_col (int): borders of the subset of teh interest
    Returns:
    - tuple: Two xarray DataArrays representing the ensemble averages of the ice_u and ice_v 
             components for each hour.
    """
    
    # Initialize lists to store averaged data
    avg_ice_u = []
    avg_ice_v = []
    
    #jt = JohannesroThredds()
    
    for time in time_period:
        data = jt.get_var_data(time, min_row, max_row, min_col, max_col )
    
        # Average the data over 24 ensembles andappend to lists
        avg_ice_u.append(np.nanmean(data['ice_u'], axis=0))
        avg_ice_v.append(np.nanmean(data['ice_v'], axis=0))
        
        
    return avg_ice_u, avg_ice_v


def cumulative_ice_displacement(X, Y, x, y, ice_u, ice_v, time_period, time_diff_start, time_diff_end):
    """
    Computes the integrated displacement along the x and y axes for each hour (or its fraction) of the forecasting  time period. 
    The calculation for each hour is cumulative. For instance, the displacement value at the 1st hour represents the displacement 
    for that hour alone, while the value at the 5th hour represents the cumulative displacement over the first five hours.
    Such displacmeents are intended for drift-driven warping
    
    This function also derives the hourly coordinates (xx and yy) in the Lagrangian reference frame, where the model coordinates 
    X and Y serve as the starting grid coordinates.

    Parameters:
    - X, Y (DataArray): Cartesian coordinates from the model or its subset.
    - ice_u, ice_v (DataArray): Ice velocities along the x and y axes, respectively.
    - time_period (DataArray): The input xarray model time period.
    - time_diff_start (int): Time difference in seconds between SAR1 and the start of the model time period.
    - time_diff_end (int): Time difference in seconds between SAR2 and the end of the model time period.

    Returns: 
    - xx, yy (list): Lists of coordinates corresponding to each timestamp in the time period. The lengths of xx and yy are equivalent to len(time_period).
    - int_dx, int_dy (list): Lists of total cumulative displacements for each hour of the time period. The lengths of int_dx and int_dy are len(time_period) - 1.
    """


    # Create coordinate arrays  
    x_sub, y_sub = x, y
    # Store the shape of the original arrays
    shape = x.shape
    # Turn arrays into vectors
    x_sub = x_sub.flatten()
    y_sub = y_sub.flatten()

    # Store the selected coordinate arrays in lists to collect arrays from different iterations for further processing
    xx = [x_sub]
    yy = [y_sub]

    # Prepare lists for integrated drift for comparison with model
    int_dx = []
    int_dy = []

    for t in range(1, len(time_period)):
        #print(f'{t} hour done')

        # Clip y_stp and x_stp to ensure they are within the bounds of the coordinate grid
        #y_stp = np.clip(y_stp, np.min(Y_stp.data), np.max(Y_stp.data))
        #x_stp = np.clip(x_stp, np.min(X_stp.data), np.max(X_stp.data))

        # Extract instantaneous velocity data for the hour start and end using polygon mask for area of interest
        u1 = ice_u[t-1]
        u2 = ice_u[t]

        v1 = ice_v[t-1]
        v2 = ice_v[t]

        # Calculate displacement along lon (u) and lat (v) based on mean velovity
        # The first and the last displacemnets are calculated differently based on time difference with the SAR acquisitions
        if t == 1:
            u_displacement = (u2+u1)*(3600-time_diff_start)/2
            v_displacement = (v2+v1)*(3600-time_diff_start)/2
            #print("start", t, u_displacement.values)
        elif t in range(2, len(time_period)-1):
            u_displacement = (u2+u1)*3600/2
            v_displacement = (v2+v1)*3600/2
            #print(t, u_displacement.values)
        elif t == len(time_period)-1:
            u_displacement = (u2+u1)*(3600-time_diff_end)/2
            v_displacement = (v2+v1)*(3600-time_diff_end)/2
            #print("end", t, u_displacement.values)

        # Create interpolator object with grid coordinates and corresponded drift values
        ut_interpolator = RegularGridInterpolator((Y,X), u_displacement, bounds_error=False, fill_value = None)
        dx = ut_interpolator((y_sub, x_sub))
        #print("dx", dx)
        vt_interpolator = RegularGridInterpolator((Y,X), v_displacement, bounds_error=False, fill_value = None)
        dy = vt_interpolator((y_sub, x_sub)) 
        #print("dy", dy)
        # Calculate new coordinates based on displacement 
        x_sub = x_sub + dx
        y_sub = y_sub + dy
        # Append them to the list
        xx.append(x_sub)
        yy.append(y_sub)


        #Calculate total integrated drift
        if t == 1:
            dx_total = dx
            dy_total = dy
            int_dx.append(dx_total)
            int_dy.append(dy_total)
        else:
            dx_total = int_dx[-1] + dx
            dy_total = int_dy[-1] + dy
            int_dx.append(dx_total)
            int_dy.append(dy_total)
            
    return xx, yy, int_dx, int_dy


def plot_model_drift_results(sar_drift_output_path, x, y, upm, vpm, sar_disp_min, sar_disp_max):
    u = model_u / 1000  # convert to kilometers
    v = model_v / 1000  # convert to kilometers
    disp = np.sqrt((v**2 + u**2))  # calculate displacement magnitude

    # Close any existing figures
    plt.close('all')

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    quiv_params = {
        'scale': 900,
        'cmap': 'jet',
        'width': 0.002,
        'headwidth': 3,
        'clim': (sar_disp_min, sar_disp_max)
    }

    step = 1
    # Create quiver plot
    quiv = ax.quiver(x[::step], y[::step], u[::step], v[::step], disp[::step], **quiv_params)

    # Colorbar with shared color scale
    cbar = fig.colorbar(quiv, ax=ax, orientation='vertical', shrink=0.9)
    cbar.set_label('Displacement Magnitude [km]')

    # Set the x and y limits for the axis
    ax.set_xlim([300000, 600000])
    ax.set_ylim([150000, 600000])

    # Title of the plot
    ax.set_title(f'Barents2.5 Model Ice Drift Displacement ')

    # Set background color to white
    fig.set_facecolor('white')

    # Adjust layout for tight fit
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    # Save the figure without displaying it
    save_path = os.path.join(sar_drift_output_path, f"Model_drift_field.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)