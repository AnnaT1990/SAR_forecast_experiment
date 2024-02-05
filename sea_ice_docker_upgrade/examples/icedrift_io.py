#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------- #
# =========================== ICEDRIFT I/O ================================== #
# --------------------------------------------------------------------------- #
r"""
    icedrift/icedrift_io.py
    << description >> 
    Reading/writing functionality. 
    
    << structure >>
    - Imports and settings.
    - Using rasterio for reading geotiffs. Currently the only accepted input.
    - Using hdf5 for creating NetCDF output files. 
    - Using rasterio for writing geotiffs.     
    
TODO: io/reading - read lat-lon files
TODO: io/writing - implement lat-lon writing
TODO: io/writing - use pyproj to enable user-determined projection (requires extra input option)
TODO: io/writing - set software VERSION parameter as output parameter

"""
#
# Author: Jelte van Oostveen
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------#
# ============================ IMPORTS & SETTINGS =============================#
# -----------------------------------------------------------------------------#

import rasterio
import pyproj
from netCDF4 import Dataset
from loguru import logger
from numpy import NaN, float32, save, meshgrid, zeros_like
from icedrift.icedrift_utils import nan_to_zero

import sys
import datetime
import pathlib
import warnings

# -----------------------------------------------------------------------------#
# ================================= READING ===================================#
# -----------------------------------------------------------------------------#

def check_geotiff_singleband(dataset):
    """
    Checks if geotiff is singleband.

    Reads location and filename of a singleband geotiff,
    checks is the geotiff is really singleband, 
    exits with error otherwise 

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Must be open to read contents.
    
    Returns
    -------
    Boolean
        Returns True only when the input geotiff is singleband.
        Program exits with code 1 otherwise.
    """
    # check if really singleband in metadata, otherwise throw error
    if len(dataset.indexes) == 1:
        logger.debug("OK. Singleband dataset found.")
    else:
        logger.error(
            "Multiband dataset: {} found, while only Singleband datasets are accepted. Please check",
            dataset.name,
        )
        sys.exit(1)


def check_for_equal_projection(dataset1, dataset2):
    """
    Checks if two datasets have the same projection.
    
    Rasterio CRS information is directly compared.

    Parameters
    ----------
    dataset1 : rasterio.io.DatasetReader
        Must be open to read contents.
    dataset2 : rasterio.io.DatasetReader
        Must be open to read contents.

    Returns
    -------
    Boolean
        Returns True when two datasets have the same projection. 
        Program exits with code 1 otherwise.
    """
    equal_proj = False

    if dataset1.crs == None and dataset2.crs == None:
        logger.warning("No projection information found. Continuing without...")
        equal_proj = True
        return equal_proj

    if dataset1.crs == dataset2.crs:
        logger.debug(f"Dataset 1: {dataset1.name} and dataset 2: {dataset2.name} have the same projection.")
        logger.debug("{}".format(dataset1.crs))
        equal_proj = True
        return equal_proj
    else:
        logger.error(
            f"Dataset 1: {dataset1.name} and dataset 2: {dataset2.name} do not have the same projection. Please check"
        )
        return equal_proj
        sys.exit(1)


def open_dataset(dataset_fname, suppress_warning=False):
    """
    Opens a dataset with Rasterio.

    Expecting either GeoTiff (.tif) or ENVI-style binary (.img) 

    Parameters
    ----------
    dataset_fname : pathlib.Path
        Containing path and filename of input dataset.
    suppress_warning : bool, optional
        If True, any warnings caused by the lack of a GeoTransform 
        and/or a CRS in the input dataset are suppressed. By default False

    Returns
    -------
    rasterio.io.DatasetReader
        Openend.
    """
    # Rasterio throws an ugly warning when there's no geo information
    # available in the input dataset, hence the Warning-catching.
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try: 
            dataset = rasterio.open(dataset_fname)
        except Warning as w:
            if suppress_warning == False:
                logger.warning(f'{w}. Continuing without geoinfo.')
                logger.warning(f'This might have implications for the dataset resolution when reading pyramids. Check carefully!')
                logger.warning(f'Dataset without geotransform: {dataset_fname}')

        warnings.filterwarnings('ignore')           
        dataset = rasterio.open(dataset_fname)

    if dataset_fname.suffix.lower() == '.tif':
        check_geotiff_singleband(dataset)
    return dataset


def close_dataset(dataset):
    """
    Close an open Rasterio dataset.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Openend.
    """
    if type(dataset) == rasterio.io.DatasetReader:
        dataset.close()


# ---
# READING DATA
def read_block(dataset, ul_indices, block_shape, band=1):
    """
    Read a chunck from a dataset using a Rasterio Window.

    Rasterio windows are read from upper left (UL): 
    rasterio.window.Window(col_off, row_off, width, height)
    Where col_off, row_off are the pixel offsets from the UL of the dataset
    to the UL of the Window.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Must be open to read contents.
    ul_indices : list
        Containing upper left (UL) coordinates of Rasterio window:
        [row, col] or [y, x].
    block_shape : list
        Containing size of Rasterio Window [width, height].

    Returns
    -------
    numpy.ndarray
        Contains float32 image data within Window bounds.
    """
    
    window = rasterio.windows.Window(
        ul_indices[1], ul_indices[0], block_shape[0], block_shape[1]
    )
    data = dataset.read(band, window=window, boundless=True, fill_value=NaN)
    return data.astype(float32)


def read_overview(dataset, overview_level):
    """
    Read overview according to specific overview level.

    This means in practice: scale the size of the image to match to overview level.
    For reading of (internal- or external) overviews, please refer to:
    https://gis.stackexchange.com/questions/353794/decimated-and-windowed-read-in-rasterio

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Must be open to read contents.
    overview_level : int
        Indicating overview level: 4.

    Returns
    -------
    numpy.ndarray
       Contains float32 image data of dataset scaled by the overview level.
    """
    overview_shape = (
        1,
        int(dataset.height / overview_level),
        int(dataset.width / overview_level),
    )
    data = dataset.read(1, out_shape=overview_shape)
    return data.astype(float32)


def read_overview_block(dataset, ul_indices, block_shape, band=1):
    """
    Read a part of an external overview within a dataset using a Rasterio Window.

    NOTE that the Window is not read from an external overview file (.ovr), 
    but rather from an reduced-resolution on-disk copy of the original dataset.
    The reason behind this is that reading a Window from overview file with 
    options: 'Boundless = True' and 'fill_value = NaN' does not work when reading
    from GDAL/Rasterio overviews directly.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Must be open to read contents.
    ul_indices : list
        Containing upper left (UL) coordinates of Rasterio window:
        [row, col] or [y, x].
    block_shape : list
        containing size of Rasterio window to be read [height, width].
    band : int, optional
        [description], by default 1.

    Returns
    -------
    numpy.ndarray
        Contains float32 image data of dataset
    """
    return read_block(dataset, ul_indices, block_shape, band=band)


def centercoords_to_ulindices(dataset, center_coordinates, block_shape):
    """
    Convert center coordinates to upper left indices 
    for a block read from a Rasterio dataset.

    The upper left indices are found by subracting 1/2 the requested 
    block shape from the center coordinates, and subsequently using these 
    corner coordinates to extract the row-, col-indices from the dataset.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Must be open to read contents.
    center_coordinates : list
        [x, y] coordinates in (projected) coordinate system.
    block_shape : list
        Shape of block to consider: [height, width]

    Returns
    -------
    list
        row, col (dataset indices of UL corner of a block with shape:
        'block_shape' and center position: 'center_coordinate')
    """
    # Subtraction because low to high values increase from left to right
    ulx = center_coordinates[0] - (block_shape[1] // 2) * dataset.res[0]

    # Addition because low to high values decrease from top to bottom
    uly = center_coordinates[1] + (block_shape[0] // 2) * dataset.res[1]
    
    # row, col = dataset.index(x, y)
    return dataset.index(ulx, uly)


def extract_block (dataset, center_coordinates, block_shape, apriori=0, shift=None, pyramid_level=None):
    """
    Reads a chunck from a Rasterio dataset around a set of input coordinates.

    In the simplest case, the input center coordinates will be converted to
    upper left (UL) coordinates, as the Rasterio Window-read function expects UL
    coordinates to read from a file. 
    In the case of a-priori information, a shift is expected and added to the UL corner 
    coordinates. The shift is then scaled by the current pyramid-level as the 'apriori' 
    switch is also used cascaded/pyramid processing where an earlier iteration found a 
    valid shift. 

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Must be open to read contents.
    center_coordinates : list
        [x, y] coordinates in (projected) coordinate system.
    block_shape : list
        Shape of block to consider: [height, width].
    apriori : int, optional
        Switch for a-priori displacement input, by default 0.
        If 1, optional arguments 'shift' and 'pyramid_level' are required.
    shift : list, optional
        Contains a shift in x and y-direction [dx, dy], by default None.
        Expected when 'apriori' == 1.
    pyramid_level : int, optional
        Expected when 'apriori' == 1, by default None.

    Returns
    -------
    numpy.ndarray
        Block of size 'block_shape' centered on 'center_coordinates' 
        read from a Rasterio 'dataset'.

    Raises
    ------
    ValueError
        If the a-priori switch is used, the two other optional input
        parameters 'shift' and 'pyramid_level' are expected.
    """
    row, col = centercoords_to_ulindices(
        dataset, center_coordinates, block_shape
    )

    if apriori == 1:
        if shift != None and pyramid_level != None:
            col += (shift[0] / pyramid_level)
            row -= (shift[1] / pyramid_level)
        else:
            raise ValueError(f"Wrong entry for: {shift}, or: {pyramid_level}")

    ulindices = row, col

    block = read_overview_block(
        dataset, ulindices, block_shape
    )
    
    return nan_to_zero(block)


def sample_dataset (dataset, coordinates):
    """
    Extract values from a Rasterio dataset on specific coordinates.

    Uses rasterio.io.DatasetReader.sample() to extract values at a range of 
    coordinates.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Must be open to read contents.
    coordinates : list
        Contains two lists: [x_coordinates, y_coordinates]. See for specifics:
        utils.create_coordinate_vectors().

    Returns
    -------
    numpy.ndarray
        Contains values sampled from 'dataset' at 'coordinates'.
    """
    logger.info(f"Sampling apriori dataset {dataset.name}")

    x_coordinate_grid, y_coordinate_grid = meshgrid(
        coordinates[0], coordinates[1]
    )
    coordinate_pairs = zip(
        x_coordinate_grid.flatten(), 
        y_coordinate_grid.flatten()
    )

    values = zeros_like(x_coordinate_grid.flatten())
    for i, p in enumerate(coordinate_pairs):
        values[i] = list(dataset.sample([p]))[0][0]

    return values.reshape(x_coordinate_grid.shape)


def to_pixel_displacements(array, resolution, time_gap=0):
    """
    Converts input velocity [m/d] or displacement [m], to displacements in pixels.
    
    When a time gap is entered, the input values are multiplied by the time gap 
    and divided by the dataset resolution of the input datasets. When no time is 
    given, the input values are only divided by the resolution of the input 
    datasets, to convert any output to pixel offsets.

    Parameters
    ----------
    array : numpy.ndarray
        Contains displacement data [m] or velocity [m/d].
    resolution : int or float
        Dependent on which array gets scaled, input the x- or y resolution of 
        the input full resolution datasets.
    time_gap : float
        Indicates the temporal baseline between the two input images

    Returns
    -------
    numpy.ndarray
        Contains pixel displacements
    """ 
    if time_gap:
        return array / resolution * time_gap
    else:
        return array / resolution


# -----------------------------------------------------------------------------#
# ================================= WRITING ===================================#
# -----------------------------------------------------------------------------#

def _set_netcdf_global(ncfile, params):
    """
    Adds global attributes to an open NetCDF file.

    *PRIVATE*

    Parameters
    ----------
    ncfile : netCDF4._netCDF4.Dataset
        Must be open to write contents.
    params : dict
        Icedrift processing parameters.
    """
    # Global attributes (belonging to entire dataset)
    ncfile.institution = "UiT The Arctic University of Norway - Centre for Integrated Remote Sensing and Forecasting for Arctic Operations (CIRFA)"
    ncfile.source = "CIRFA Icedrift"
    ncfile.history = "Created: {}".format(datetime.datetime.now())
    ncfile.comment = ("User-defined settings for this run are available within group 'parameters'")
    ncfile.Conventions = "CF-1.8"
    ncfile.license = "No restrictions on access or use of data generated by this research code. Other restrictions may apply depending on input data."

    # Include all user-defined settings in NetCDF
    # can also be added in a group, but they won't be visible by gdalinfo
    # might try to create a separate variable: parameters?
    parameters = ncfile.createGroup("parameters")
    for k, v in params.items():
        try:
            setattr(parameters, k, str(v))
            # setattr(ncfile, k, v)
        # NetCDF cannot cope with pathlib, change to str objects:
        except TypeError:
            if v.is_file():
                setattr(parameters, k, str(v.name))
            elif v.is_dir():
                setattr(parameters, k, str(v))


def write_netCDF_multiband(bands, band_names, band_descriptions, coordinates, crs, params):
    """
    Writing Icedrift result arrays to one NetCDF file with multiple bands.

    The files are written in CF-format (1.8). Any CRS information is 
    converted to projection-specific CF-format using PyProj. 

    See 'write_netCDF_singleband' for more background on the GeoTransform.

    Parameters
    ----------
    bands : list
        Containing a numpy.ndarray for each result band.
    band_names : list
        Containing str with names corresponding to 'bands'.
    band_descriptions : list
        Containing str with descriptions corresponding to 'bands'.
    coordinates : list
        Containing two 1d lists as coordinate pairs: x_coordinates, y_coordinates.
        The length of the lists is equal to the dimensions (width, height) of the
        bands.
    crs : rasterio.crs.CRS
        Coordinate reference system information. Object includes exporting options
        as: crs.to_proj() or crs.to_wkt().
    params : dict
        Icedrift processing parameters.
    """
    # NAMING
    ncfile_name = params["dest_dir"] / (params["run_name"] + ".nc")
    ncfile = Dataset(ncfile_name, "w", format="NETCDF4")

    # DIMENSIONS
    dimensions = bands[0].shape
    x = ncfile.createDimension("x", dimensions[1])
    y = ncfile.createDimension("y", dimensions[0])

    # VARIABLES
    x_var = ncfile.createVariable("x_var", "f4", ("x",))
    y_var = ncfile.createVariable("y_var", "f4", ("y",))

    variables = []
    for i in range(len(bands)):
        variables.append(
            ncfile.createVariable(f"{band_names[i]}", "f4", ("y", "x"), fill_value=0.0)
        )

    # GLOBAL ATTRIBUTES
    _set_netcdf_global(ncfile, params)
    ncfile.title = f"{ncfile_name.stem}"

    # SET PROJECTION
    try:
        # Use pyproj to convert epsg's to any CF standards
        crs_cf = pyproj.CRS(crs.to_epsg()).to_cf()
    except Exception as e:
        logger.error(e)
        logger.error("Unknown projection. Cannot use projection information for netCDF writing!")
        logger.info("Unable to export in netCDF. Escaping to singleband ENVI binary.") 
        write_ENVI_singleband (bands, band_names, band_descriptions, coordinates, crs, params)
        ncfile_name.unlink()
        return

    # Create variable to hold projection info
    projection = ncfile.createVariable(f"{crs_cf.get('grid_mapping_name')}", "S1")  
    
    # Fill projection variable with crs details from the Pyproj CF dict
    for k, v in crs_cf.items():
        setattr(projection, k, v)

    # Set geotransform for proper assignment of corner coordinates 
    x_coordinates, y_coordinates = coordinates
    transform = rasterio.transform.from_origin(
        x_coordinates[0] - 0.5 * (x_coordinates[1] - x_coordinates[0]),
        y_coordinates[0] - 0.5 * (y_coordinates[1] - y_coordinates[0]),
        x_coordinates[1] - x_coordinates[0],
        y_coordinates[0] - y_coordinates[1],
    )
    projection.GeoTransform=" ".join(str(x) for x in transform.to_gdal())

    # Update spatial reference (the automatic generated CF one does not work, hence: delete)
    projection.spatial_ref = crs.to_wkt()
    del projection.crs_wkt 

    for i, variable in enumerate(variables):
        variable.long_name = band_descriptions[i]
        variable.grid_mapping = projection.name

    # VARIABLE ATTRIBUTES
    x_var.units = "m"
    x_var.standard_name = "projection_x_coordinate"
    x_var.long_name = "Easting"

    y_var.units = "m"
    y_var.standard_name = "projection_y_coordinate"
    y_var.long_name = "Northing"

    # VARIABLES ASSIGNMENT
    x_coordinates, y_coordinates = coordinates
    x_var[:] = x_coordinates
    y_var[:] = y_coordinates
    for variable, band in zip(variables, bands):
        variable[:] = band

    # WRITING (actually writing to disk)
    ncfile.close()
    logger.info("Data saved in multiband NetCDF format to {}".format(ncfile_name))


def write_netCDF_singleband(bands, band_names, band_descriptions, coordinates, crs, params):
    """
    Writing Icedrift result arrays to separate NetCDF files.

    The files are written in CF-format (1.8). Any CRS information is 
    converted to projection-specific CF-format using PyProj. 

    The GeoTransform, which details the corner coordinates and pixel size of the 
    data, can be pre-calculated and set as part of the projection. However, the
    GeoTransform is also automatically generated when dimension variables x and
    y are present and corresponding to the names of the dimensions (as it should
    be based on CF-conventions). However, the resulting GeoTransform may then 
    contain corner coordinates and resolution sizes that are not well-rounded
    and (slightly) different from the GeoTiff output generated by Icedrift. The 
    keywords:
        'x.standard_name = "projection_x_coordinate"'
        'y.standard_name = "projection_y_coordinate"'
    are key components when GDAL or QGIS is determining the GeoTransform. When
    the GeoTransform is set manually, but the coordinate variables are listed
    too, the manual GeoTransform gets ignored, resulting in slightly incorrect 
    metadata. To prevent this from happening, the coordinate variables x, y are 
    renamed to x_var and y_var. This way, the coordinate grids are still included 
    in the output file, but the correct GeoTransform is used. 
    
    The values are succesfully tested against the 'singleband_geotiff' output.

    For more info, see:
    https://github.com/OSGeo/gdal/blob/4cdb36c4273e4949a0912c207f0383975aaef313/gdal/frmts/netcdf/netcdfdataset.cpp
    https://gis.stackexchange.com/questions/250977/write-projected-array-to-netcdf-file-best-practice 
    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#appendix-grid-mappings

    NOTE: There might be alternative solutions for this issue. However, at the time
    of writing, this is the most practical and tested solution.

    Parameters
    ----------
    bands : list
        Containing a numpy.ndarray for each result band.
    band_names : list
        Containing str with names corresponding to 'bands'.
    band_descriptions : list
        Containing str with descriptions corresponding to 'bands'.
    coordinates : list
        Containing two 1d lists as coordinate pairs: x_coordinates, y_coordinates.
        The length of the lists is equal to the dimensions (width, height) of the
        bands.
    crs : rasterio.crs.CRS
        Coordinate reference system information. Object includes exporting options
        as: crs.to_proj() or crs.to_wkt().
    params : dict
        Icedrift processing parameters.
    """
    for i in range(len(bands)):
        data = bands[i]
        data_name = band_names[i]
        data_description = band_descriptions[i]

        # NAMING
        ncfile_name = params["dest_dir"] / (params["run_name"] + f"-{data_name}.nc")
        ncfile = Dataset(ncfile_name, "w", format="NETCDF4")

        # DIMENSIONS
        x = ncfile.createDimension("x", data.shape[1])
        y = ncfile.createDimension("y", data.shape[0])

        # VARIABLES
        x_var = ncfile.createVariable("x_var", "f4", ("x",))
        y_var = ncfile.createVariable("y_var", "f4", ("y",))

        # This netcdf band relies on two dimensions (x, y)
        band = ncfile.createVariable(f"{data_name}", "f4", ("y", "x"), fill_value=0.0)
        
        try:
            # Use pyproj to convert epsg's to any CF standards
            crs_cf = pyproj.CRS(crs.to_epsg()).to_cf()
        except Exception as e:
            logger.error(e)
            logger.error("Unknown projection. Cannot use projection information for netCDF writing!")
            logger.info("Unable to export in netCDF. Escaping to singleband ENVI binary.") 
            ncfile_name.unlink()
            write_ENVI_singleband ([data], [data_name], [data_description], coordinates, crs, params)
            if i+1 == len(bands):
                return

        # Create variable to hold projection info
        projection = ncfile.createVariable(f"{crs_cf.get('grid_mapping_name')}", "S1")  
        
        # Fill projection variable with crs details from the Pyproj CF dict
        for k, v in crs_cf.items():
            setattr(projection, k, v)

        # Set geotransform for proper assignment of corner coordinates 
        x_coordinates, y_coordinates = coordinates
        transform = rasterio.transform.from_origin(
            x_coordinates[0] - 0.5 * (x_coordinates[1] - x_coordinates[0]),
            y_coordinates[0] - 0.5 * (y_coordinates[1] - y_coordinates[0]),
            x_coordinates[1] - x_coordinates[0],
            y_coordinates[0] - y_coordinates[1],
        )
        projection.GeoTransform=" ".join(str(x) for x in transform.to_gdal())

        # Update spatial reference (the automatic generated CF one does not work, hence: delete)
        projection.spatial_ref = crs.to_wkt()
        del projection.crs_wkt 

        # GLOBAL ATTRIBUTES
        _set_netcdf_global(ncfile, params)
        ncfile.title = f"{ncfile_name.stem}"

        # VARIABLE ATTRIBUTES
        x_var.units = "m"
        x_var.standard_name = "projection_x_coordinate"
        x_var.long_name = "Easting"

        y_var.units = "m"
        y_var.standard_name = "projection_y_coordinate"
        y_var.long_name = "Northing"

        band.long_name = data_description
        band.fill_value = 0

        # DATA
        # putting data into the variables
        x_var[:] = x_coordinates
        y_var[:] = y_coordinates
        band[:] = data

        # SET PROJECTION TO DATA
        band.grid_mapping = projection.name

        # WRITING
        # actually writing the dataset to disk
        ncfile.close()

        logger.info(
            "Data {} saved in singleband NetCDF format to {}".format(data_name, ncfile_name)
        )


def write_geotiff_multiband(bands, band_names, band_descriptions, coordinates, crs, params):
    """
    Writing Icedrift result arrays to one NetCDF file with multiple bands.

    Parameters
    ----------
    bands : list
        Containing a numpy.ndarray for each result band.
    band_names : list
        Containing str with names corresponding to 'bands'.
    band_descriptions : list
        Containing str with descriptions corresponding to 'bands'.
    coordinates : list
        Containing two 1d lists as coordinate pairs: x_coordinates, y_coordinates.
        The length of the lists is equal to the dimensions (width, height) of the
        bands.
    crs : rasterio.crs.CRS
        Coordinate reference system information. Object includes exporting options
        as: crs.to_proj() or crs.to_wkt().
    params : dict
        Icedrift processing parameters.
    """
    geotiff_name = params["dest_dir"] / (params["run_name"] + ".tif")
    
    x_coordinates, y_coordinates = coordinates
    transform = rasterio.transform.from_origin(
        x_coordinates[0] - 0.5 * (x_coordinates[1] - x_coordinates[0]),
        y_coordinates[0] - 0.5 * (y_coordinates[1] - y_coordinates[0]),
        x_coordinates[1] - x_coordinates[0],
        y_coordinates[0] - y_coordinates[1],
    )

    dimensions = bands[0].shape
    datatype = bands[0].dtype 

    with rasterio.open(
        geotiff_name,
        "w",
        driver="GTiff",
        height=dimensions[0],
        width=dimensions[1],
        count=len(bands),
        dtype=datatype,
        crs=crs,
        transform=transform,
        nodata=0.0,
    ) as dst:

        # Adding metadata
        for i, band_description in enumerate(band_descriptions):
            dst.set_band_description(i + 1, band_description)

        dst.update_tags(data=[band_name for band_name in band_names])
        dst.update_tags(dataset1=params["dataset1"])
        dst.update_tags(dataset2=params["dataset2"])
        dst.update_tags(time_gap=params["time_gap"])
        dst.update_tags(run_name=params["run_name"])
        dst.update_tags(cascades=len(params["cascade_levels"]))
        dst.update_tags(pyramids=len(params["pyramid_levels"]))
        dst.update_tags(min_template_shape=params["min_template_shape"])
        dst.update_tags(overlap_factor=params["overlap_factor"])
        dst.update_tags(step_size=params["stepsize"])
        dst.update_tags(cascade_levels=params["cascade_levels"])
        dst.update_tags(pyramid_levels=params["pyramid_levels"])
        dst.update_tags(min_window_shape=params["min_window_shape"])
        dst.update_tags(cores=params["cores"])
        dst.update_tags(correlation_threshold=params["correlation_threshold"])

        # Actual writing
        for i, data in enumerate(bands):
            dst.write(data, i + 1)

    logger.info("All data saved in multiband GeoTIFF format to {}".format(geotiff_name))


def write_geotiff_singleband(bands, band_names, band_descriptions, coordinates, crs, params):
    """
    Writing Icedrift result arrays to separate GeoTiff files.

    Parameters
    ----------
    bands : list
        Containing a numpy.ndarray for each result band.
    band_names : list
        Containing str with names corresponding to 'bands'.
    band_descriptions : list
        Containing str with descriptions corresponding to 'bands'.
    coordinates : list
        Containing two 1d lists as coordinate pairs: x_coordinates, y_coordinates.
        The length of the lists is equal to the dimensions (width, height) of the
        bands.
    crs : rasterio.crs.CRS
        Coordinate reference system information. Object includes exporting options
        as: crs.to_proj() or crs.to_wkt().
    params : dict
        Icedrift processing parameters.
    """
    for i in range(len(bands)):
        data = bands[i]
        data_name = band_names[i]
        data_description = band_descriptions[i]

        if data.dtype != 'float32':
            data = float32(data)

        geotiff_name = params["dest_dir"] / (params["run_name"] + f"-{data_name}.tif")

        x_coordinates, y_coordinates = coordinates
        transform = rasterio.transform.from_origin(
            x_coordinates[0] - 0.5 * (x_coordinates[1] - x_coordinates[0]),
            y_coordinates[0] - 0.5 * (y_coordinates[1] - y_coordinates[0]),
            x_coordinates[1] - x_coordinates[0],
            y_coordinates[0] - y_coordinates[1],
        )

        with rasterio.open(
            geotiff_name,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=0.0,
        ) as dst:
            # Adding metadata
            dst.set_band_description(1, data_description)
            dst.update_tags(data=data_name)
            dst.update_tags(dataset1=params["dataset1"])
            dst.update_tags(dataset2=params["dataset2"])
            dst.update_tags(time_gap=params["time_gap"])
            dst.update_tags(run_name=params["run_name"])
            dst.update_tags(cascades=len(params["cascade_levels"]))
            dst.update_tags(pyramids=len(params["pyramid_levels"]))
            dst.update_tags(min_template_shape=params["min_template_shape"])
            dst.update_tags(overlap_factor=params["overlap_factor"])
            dst.update_tags(step_size=params["stepsize"])
            dst.update_tags(cascade_levels=params["cascade_levels"])
            dst.update_tags(pyramid_levels=params["pyramid_levels"])
            dst.update_tags(min_window_shape=params["min_window_shape"])
            dst.update_tags(cores=params["cores"])
            dst.update_tags(correlation_threshold=params["correlation_threshold"])

            # Actual writing
            dst.write(data, 1)

        logger.info(
            "Data {} saved in singleband GeoTIFF format to {}".format(
                data_name, geotiff_name
            )
        )


def write_ENVI_singleband (bands, band_names, band_descriptions, coordinates, crs, params):
    """
    Writing Icedrift result arrays to separate ENVI-style binary files (no geoinfo!).

    ENVI-style binary files means that two files are saved to disk:
        - '.img'
        - '.hdr'
    Where the '.hdr' header file contains information about the binary '.img' file. 
    Icedrift processing parameters are saved as comments (';') in the header file.

    As 'write_ENVI_singleband' is the fallback option to 'write_netcdf_singleband' and
    'write_netcdf_multiband', when no- or an invalid CRS is found, the coordinate lists
    are saved as coordinate grids: x.img, x.hdr, y.img, y.hdr. This way the results can
    be georeferenced outside of Icedrift. 

    Parameters
    ----------
    bands : list
        Containing a numpy.ndarray for each result band.
    band_names : list
        Containing str with names corresponding to 'bands'.
    band_descriptions : list
        Containing str with descriptions corresponding to 'bands'.
    coordinates : list
        Containing two 1d lists as coordinate pairs: x_coordinates, y_coordinates.
        The length of the lists is equal to the dimensions (width, height) of the
        bands.
    crs : rasterio.crs.CRS
        Coordinate reference system information. Object includes exporting options
        as: crs.to_proj() or crs.to_wkt().
    params : dict
        Icedrift processing parameters.
    """
    logger.info("Saving output in singleband ENVI format (2 files: .hdr, and .img)")
    logger.info("NOTE: coordinate grids (in x and y direction separately) are additional output")
    x_coordinates, y_coordinates = coordinates
    x_coordinate_grid, y_coordinate_grid = meshgrid(x_coordinates, y_coordinates)

    bands.append(x_coordinate_grid)
    bands.append(y_coordinate_grid)
    band_names.append('x')
    band_names.append('y')
    band_descriptions.append('x coordinate grid')
    band_descriptions.append('y coordinate grid')

    for i in range(len(bands)): 
        data = bands[i]
        data_name = band_names[i]
        data_description = band_descriptions[i]

        hdr_name = params["dest_dir"] / (params["run_name"] + f"-{data_name}.hdr")
        img_name = params["dest_dir"] / (params["run_name"] + f"-{data_name}.img")

        # Determine Windows or Unix-based machines for the byte order.
        if sys.byteorder == 'little':
            byteorder = 0
        elif sys.byteorder == 'big':
            byteorder = 1
        else:
            logger.error('Endianess (byte order) unknown. Not writing to disk')
            return

        # Create metadata
        hdr_dict = {
            'description': data_description,
            'header offset' : 0,
            'file type' : 'ENVI Standard',
            'data type' : 4,
            'samples' : data.shape[1],
            'lines' : data.shape[0],
            'bands' : 1,
            'band names' : data_name,
            'byte order' : byteorder,
            'interleave' : 'bsq',
        }

        # Write HDR file
        with open(hdr_name, 'w') as write_hdr: 
            write_hdr.write('ENVI\n')
            for key, value in hdr_dict.items():
                write_hdr.write(f'{key} = {value}\n')
             
            # Add processing parameters as comments
            write_hdr.write('\n;Icedrift processing parameters below:\n')
            for key, value in params.items():
                write_hdr.write(f';{key} = {value}\n')

        # Write binary IMG file
        data.tofile(img_name)

        logger.info(
            "Data {} saved in singleband ENVI format to {}".format(
                data_name, hdr_name,
            )
        )


'''
def write_netcdf_latlon_singleband (data, data_name, data_description, coordinates, crs, params):
    """
        TODO: implement a working version of write_netcdf_latlon_singleband()
    """
    dataset = Dataset(dataset_str, 'w', format='NETCDF4_CLASSIC')

    # create dimensions
    lat = dataset.createDimension('lat', 73)
    lon = dataset.createDimension('lon', 144)

    # create coordinate variables for 2 dimensions    
    latitudes = dataset.createVariable('latitude', 'f4', ('lat',))
    longitudes = dataset.createVariable('longitude', 'f4', ('lon',))

    # global attributes 
    _set_netcdf_global (ncfile, params)

    # putting data into the variables - example
    lats = np.arange(-90,91,2.5)
    lons = np.arange(-180,180,2.5)
    latitudes[:] = lats
    longitudes[:] = lons

    # actually writing the dataset to disk
    dataset.close()
'''


def write(bands, band_names, band_descriptions, coordinates, crs, parameters, output_format):
    """
    Switcher for different writing options. 

    Parameters
    ----------
    bands : list
        Containing a numpy.ndarray for each result band.
    band_names : list
        Containing str with names corresponding to 'bands'.
    band_descriptions : list
        Containing str with descriptions corresponding to 'bands'.
    coordinates : list
        Containing two 1d lists as coordinate pairs: x_coordinates, y_coordinates.
        The length of the lists is equal to the dimensions (width, height) of the
        bands.
    crs : rasterio.crs.CRS
        Coordinate reference system information. Object includes exporting options
        as: crs.to_proj() or crs.to_wkt().
    params : dict
        Icedrift processing parameters.
    output_format : str
        See 'Returns' for options.

    Returns
    -------
    function
        Dependent on the 'output_format' a different function is called:
            'singleband_geotiff'    write_geotiff_singleband
            'multiband_geotiff'     write_geotiff_multiband
            'singleband_netcdf'     write_netCDF_singleband
            'multiband_netcdf'      write_netCDF_multiband
            'singleband_binary'     write_ENVI_singleband        
    """
    if output_format == None:
        logger.info('No output will be written')
        return
    else:
        switcher = {
            'singleband_geotiff': write_geotiff_singleband,
            'multiband_geotiff': write_geotiff_multiband,
            'singleband_netcdf': write_netCDF_singleband,
            'multiband_netcdf': write_netCDF_multiband,
            'singleband_binary': write_ENVI_singleband,
        }
        # Get the function from switcher dictionary
        writing_method = switcher.get(output_format)
        # Execute the function
        return writing_method(bands, band_names, band_descriptions, coordinates, crs, parameters)