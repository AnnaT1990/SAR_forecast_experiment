from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import os
import csv
from osgeo import osr



def round_start_time(t):
    """
    Rounds down the given timestamp to the nearest hour.

    Parameters:
    - t (str): The input timestamp in the format '%Y-%m-%dT%H:%M:%S'.

    Returns:
    - str: The rounded timestamp.
    """
    t_dt = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S')
    rounded_dt = t_dt.replace(minute=0, second=0)
    return rounded_dt.strftime('%Y-%m-%dT%H:%M:%S')

def round_end_time(t):
    """
    Rounds up the given timestamp to the nearest hour.

    Parameters:
    - t (str): The input timestamp in the format '%Y-%m-%dT%H:%M:%S'.

    Returns:
    - str: The rounded timestamp.
    """
    t_dt = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S')
    rounded_dt = t_dt + timedelta(hours=1)
    rounded_dt = rounded_dt.replace(minute=0, second=0)
    return rounded_dt.strftime('%Y-%m-%dT%H:%M:%S')


def time_difference(t_sar1, t_sar2, time_period):
    import numpy as np
    """
    Calculates the time differences between SAR1 and SAR2 images timestamps and the model timestamps 
    at the beginning and end of the time period. This will be used to calculate the fraction 
    of the hourly model displacement in the first and last partial hour of the forecast.

    Parameters:
    - t_sar1 (str): The timestamp for SAR1 in the format '%Y-%m-%dT%H:%M:%S'.
    - t_sar2 (str): The timestamp for SAR2 in the format '%Y-%m-%dT%H:%M:%S'.
    - time_period (DataArray): The input xarray model time period.

    Returns: 
    - time_diff_start (int): Time difference between SAR1 and the start of the model time period in seconds.
    - time_diff_end (int): Time difference between SAR2 and the end of the model time period in seconds.
    - total_time_diff (int): Total time difference between SAR1 and SAR2 images in seconds.
    """
    
    t_sar1 = np.datetime64(t_sar1)
    model_hour_start = time_period[0].values
    time_diff_start = np.timedelta64(t_sar1 - model_hour_start, 's').astype('timedelta64[s]').astype(int)
    
    t_sar2 = np.datetime64(t_sar2)
    model_hour_end = time_period[-1].values
    time_diff_end = np.timedelta64(model_hour_end - t_sar2, 's').astype('timedelta64[s]').astype(int)
    
    total_time_diff = np.timedelta64(t_sar2 - t_sar1, 's').astype('timedelta64[s]').astype(int)
    
    print(f'Time difference between SAR1 and the start of the model time period is {time_diff_start} seconds ({np.around(time_diff_start/60, 3)} minutes).') 
    print(f'Time difference between SAR2 and the end of the model time period is {time_diff_end} seconds ({np.around(time_diff_end/60, 3)} minutes).')
    print(f'Total time difference between SAR1 and SAR2 images is {total_time_diff} seconds ({np.around(total_time_diff/86400, 3)} days).') 
    
    return time_diff_start, time_diff_end, total_time_diff



def cumulative_ice_displacement(X, Y, ice_u, ice_v, time_period, time_diff_start, time_diff_end):
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
    x_sub, y_sub = np.meshgrid(X, Y)
    # Turn arrays into vectors
    x_sub = x_sub.flatten()
    y_sub = y_sub.flatten()

    # Store the selected coordinate arrays in lists to collect arrays from different iterationsfor further processing
    xx = [x_sub]
    yy = [y_sub]

    #Prepare lsit for integrated drift for comparison with model
    int_dx = []
    int_dy = []

    for t in range(1, len(time_period)):
        print(f'{t} hour done')

        # Clip y_stp and x_stp to ensure they are within the bounds of the coordinate grid
        #y_stp = np.clip(y_stp, np.min(Y_stp.data), np.max(Y_stp.data))
        #x_stp = np.clip(x_stp, np.min(X_stp.data), np.max(X_stp.data))

        # Extract instantaneous velocity data for the hour start and end using polygon mask for area of interest
        u1 = ice_u.sel(time=time_period[t-1])#.where(finalMask>0)
        u2 = ice_u.sel(time=time_period[t])#.where(finalMask>0)

        v1 = ice_v.sel(time=time_period[t-1])#.where(finalMask>0)
        v2 = ice_v.sel(time=time_period[t])#.where(finalMask>0)

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
        ut_interpolator = RegularGridInterpolator((Y.data,X.data), u_displacement.data, bounds_error=False, fill_value = None)
        dx = ut_interpolator((y_sub, x_sub))
        #print("dx", dx)
        vt_interpolator = RegularGridInterpolator((Y.data,X.data), v_displacement.data, bounds_error=False, fill_value = None)
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

def non_cumulative_ice_displacement(X, Y, ice_u, ice_v, time_period, time_diff_start, time_diff_end):
    """
     Computes average displacement along the x and y axes for each hour (or its fraction) of the forecasting time period.
    The calculation for each hour is non-cumulative: the displacement value at each hour represents the average 
    displacement for that specific hour, calculated as the mean value between two instantaneous values at the start 
    and end of that hour (or its fraction). Such displacements are intended for image-driven warping.

    The initial coordinates remain constant for each hour like in an Eulerian reference frame where the model coordinates 
    X and Y act as a fixed grid.

    Parameters:
    - X, Y (DataArray): Cartesian coordinates from the model or a subset.
    - ice_u, ice_v (DataArray): Ice velocities along the x and y axes, respectively.
    - time_period (DataArray): The input xarray model time period.
    - time_diff_start (int): Time difference in seconds between SAR1 and the start of the model time period.
    - time_diff_end (int): Time difference in seconds between SAR2 and the end of the model time period.

    Returns: 
    - xx, yy (list): Lists of coordinates corresponding to each timestamp in the time period. The lengths of xx and yy are equivalent to len(time_period).
    - hourly_dx, hourly_dy (list): Lists of non-cumulative displacements for each hour of the time period. The lengths of hourly_dx and hourly_dy are  
    len(time_period) - 1.
    """
    
    
    # Create coordinate arrays  
    x_sub, y_sub = np.meshgrid(X, Y)
    # Turn arrays into vectors
    x_sub = x_sub.flatten()
    y_sub = y_sub.flatten()

    # Store the selected coordinate arrays in lists to collect arrays from different iterationsfor further processing
    xx = [x_sub]
    yy = [y_sub]

    #Prepare lsit for integrated drift for comparison with model
    hourly_dx = []
    hourly_dy = []

    for t in range(1, len(time_period)):
        print(f'{t} hour done')

        # Extract instantaneous velocity data for the hour start and end using polygon mask for area of interest
        u1 = ice_u.sel(time=time_period[t-1])#.where(finalMask>0)
        u2 = ice_u.sel(time=time_period[t])#.where(finalMask>0)

        v1 = ice_v.sel(time=time_period[t-1])#.where(finalMask>0)
        v2 = ice_v.sel(time=time_period[t])#.where(finalMask>0)

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

        # Append hourly displacements
        dx = u_displacement.values.flatten()
        dy = v_displacement.values.flatten()
        hourly_dx.append(dx)
        hourly_dy.append(dy)
        
        # Calculate new coordinates based on displacement 
        x_sub = xx[0] + dx
        y_sub = yy[0] + dy
        # Append them to the list
        xx.append(x_sub)
        yy.append(y_sub)
                    
    return xx, yy, hourly_dx, hourly_dy


def displacement_dxdy_to_csv(output_dir, int_lon, int_lat, int_dx, int_dy, time_period):
    """
   
    Writes displacement data into multiple CSV files, each corresponding to hourly intervals.
    The function uses displacements in meters along the x (dx) and y (dy) axes as drift parameters.

    
    Parameters:
    - output_dir (str): The directory where the CSV files will be saved.
    - int_lon, int_lat (list): Lists of longitude and latitude coordinates at every time stamp of forecasting period.
    - int_dx, int_dy (list): Lists of displacement values in the x and y directions.
    - time_period (DataArray): An array representing the time points corresponding to the displacement data.
    """
    
    folder_name = os.path.join(output_dir, f'drift_du_dv')
    if not os.path.exists(folder_name):
        os.mkdir(folder_name) 

    for t in range(1,len(int_lat)):
        points=zip(int_lon[0], int_lat[0], int_dx[t-1], int_dy[t-1])
        header=['lon1','lat1', 'du','dv']

        #Create CSV file for every displacement
        time_index = pd.to_datetime(time_period[t].values)
        filename_date = time_index.strftime('%Y%m%dT%H')

        file_name = f'hourly_ice_displacement_{filename_date}_dxdy.csv'
        file_path = os.path.join(folder_name, file_name)

        #Open the CSV file for writing              
        with open(file_path, 'w', newline='') as csv_file:
            out = csv.writer(csv_file, delimiter=',')
            out.writerow(header)
            out.writerows(points)


def displacement_lon2lat2_to_csv(output_dir, int_lon, int_lat, time_period):
    """  
    Writes displacement data into multiple CSV files, with each file representing hourly data.
    The function uses destination point coordinates (lon1, lat2) in degrees as drift parameters.

    Parameters:
    - output_dir (str): The directory where the CSV files will be saved.
    - int_lon, int_lat (list): Lists of longitude and latitude coordinates at every time stamp of forecasting period.
    - time_period (DataArray): An array representing the timestamps corresponding to each set of displacement data.
    """
    
    folder_name = os.path.join(output_dir, f'drift_lon2_lat2')
    if not os.path.exists(folder_name):
        os.mkdir(folder_name) 

    for t in range(1,len(int_lat)):
        points=zip(int_lon[0], int_lat[0], int_lon[t], int_lat[t])
        header=['lon1','lat1', 'lon2','lat2']

        #Create CSV file for every displacement
        time_index = pd.to_datetime(time_period[t].values)
        filename_date = time_index.strftime('%Y%m%dT%H')

        file_name = f'hourly_ice_displacement_{filename_date}_lon2lat2.csv'
        file_path = os.path.join(folder_name, file_name)

        #Open the CSV file for writing                   
        with open(file_path, 'w', newline='') as csv_file:
            out = csv.writer(csv_file, delimiter=',')
            out.writerow(header)
            out.writerows(points)
            

        
def transform_to_geographic_coordinates(xx, yy, model_proj4_string):
    """
    Transforms coordinates from a given projection (defined by the model_proj4 string) 
    to the Geographic Coordinate System (longitude and latitude).

    Parameters:
    - xx, yy (list of np.array): Lists of x and y coordinates in the model's projection corresponding to timestamps in the forecasting period.
    - model_proj4_string (str): A Proj4 string defining the model's coordinate system.

    Returns: 
    - lon_list, lat_list (list of np.array): Lists of transformed longitude and latitude coordinates for each time stamp.
    """

    # Define the target spatial reference (WGS 84)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)

    # Define the source spatial reference (model's projection)
    model_srs = osr.SpatialReference()
    model_srs.ImportFromProj4(model_proj4_string)

    # Create the coordinate transformation
    transformation = osr.CoordinateTransformation(model_srs, target_srs)

    lon_list = []
    lat_list = []

    for x, y in zip(xx, yy):
        # Stack the coordinates for transformation
        points = np.stack((x, y), axis=1)
        
        # Apply the coordinate transformation
        transformed_points = transformation.TransformPoints(points)
        
        # Extract longitude and latitude
        lat = np.array([point[0] for point in transformed_points])
        lon = np.array([point[1] for point in transformed_points])

        lon_list.append(lon)
        lat_list.append(lat)

    return lon_list, lat_list