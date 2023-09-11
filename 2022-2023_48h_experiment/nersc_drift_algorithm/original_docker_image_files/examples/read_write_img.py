# ---- This is <read_write_img.py> ----

"""
Read and write image files using gdal.
"""

import os
import sys
import pathlib
from loguru import logger

import numpy as np
from osgeo import gdal

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

def get_all_bands(f):
  """Read all bands from input image

  Parameters
  ----------
  f: path to input img file

  Returns
  -------
  img: all bands in image file as array
  """

  # open img file
  img_file = gdal.Open(f)

  logger.debug(f'Opened img file: {f}')

  # get nuymber of bands and original data type
  n_bands       = img_file.RasterCount
  img_data_type = img_file.GetRasterBand(1).DataType

  logger.debug(f'n_bands:       {n_bands}')
  logger.debug(f'img_data_type: {img_data_type}')

  # read file as array
  img = img_file.ReadAsArray()

  logger.debug(f'Read all bands ({n_bands}) from input file')

  return img

# -------------------------------------------------------------------------- # 
# -------------------------------------------------------------------------- # 

def write_tif(f, X, as_type=gdal.GDT_Float32, overwrite=False):
  """Write array (single or multiple bands) to tif file

  Parameters
  ----------
  f : output tif file
  X : input array
  as_type : gdal data type (default: gdal.GDT_Float32)
  overwrite : overwrite if file already exists (default=False)

  Returns
  -------
  new_file : return True/False if new tif file has been created
  """

  # check if file already exists
  if os.path.exists(f):
    if overwrite==True:
      logger.debug('File already exists, deleting old file')
      os.remove(f)
    elif overwrite==False:
      logger.info('Output file already exists')
      logger.info('Exiting without writing')
      logger.info("Set overwrite option to 'True' to force new file")
      new_file = False
      return new_file

  # get dimensions and number of bands
  dims = X.shape
  if np.size(dims) == 2:
    Ny, Nx  = X.shape
    n_bands = 1
  elif np.size(dims) == 3:
    n_bands, Ny, Nx = X.shape
    if n_bands > Ny or n_bands > Nx:
      logger.warning('Number of bands is larger than number of pixels')
      logger.warning('Expected shape of input array is (n_bands, Nx, Ny)')
      logger.warning('Exiting without writing')
      new_file = False
      return new_file
  else:
    logger.error(f'Cannot write array with shape {dims} to tif file')
    logger.error('Exiting without writing')
    new_file = False
    return new_file

  logger.debug(f'n_bands: {n_bands}')
  logger.debug(f'Nx: {Nx}')
  logger.debug(f'Ny: {Ny}')
  logger.debug(f'as_type: {as_type}')

  # get driver
  output = gdal.GetDriverByName('GTiff').Create(f, Nx, Ny, n_bands, as_type)

  # write to file
  if n_bands == 1:
    output.GetRasterBand(1).WriteArray(X)
  elif n_bands > 1:
    for b in np.arange(n_bands):
      output.GetRasterBand(int(b+1)).WriteArray(X[b,:,:])

  output.FlushCache()

  new_file = True
  logger.debug('Wrote input array to tif file')

  return new_file

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def write_img(f, X, as_type=gdal.GDT_Float32, overwrite=False):
  """Write array (single band) to img file (ENVI format)

  Parameters
  ----------
  f : output img file (ENVI format)
  X : input array
  as_type : gdal data type (default: gdal.GDT_Float32)
  overwrite : overwrite if file already exists (default=False)

  Returns
  -------
  new_file : return True/False if new img file has been created

  Examples
  --------
  write_img('output_file.img', array, as_type=gdal.GDT_Float32)
  """

  # check if file already exists
  if os.path.exists(f):
    if overwrite==True:
      logger.debug('File already exists, deleting old file')
      os.remove(f)
      os.remove(os.path.splitext(f)[0]+'.hdr')
    elif overwrite==False:
      logger.info('Output file already exists')
      logger.info('Exiting without writing')
      logger.info("Set overwrite option to 'True' to force new file")
      new_file = False
      return new_file

  # get dimensions and number of bands
  dims = X.shape
  if np.size(dims) == 2:
    Ny, Nx  = X.shape
    n_bands = 1
  else:
    logger.error(f'Cannot write array with shape {dims} to img file')
    logger.error('Exiting without writing')
    new_file = False
    return new_file

  logger.debug(f'n_bands: {n_bands}')
  logger.debug(f'Nx: {Nx}')
  logger.debug(f'Ny: {Ny}')
  logger.debug(f'as_type: {as_type}')

  # get driver
  output = gdal.GetDriverByName('Envi').Create(f, Nx, Ny, n_bands, as_type)

  # write to file
  output.GetRasterBand(1).WriteArray(X)

  output.FlushCache()

  new_file = True
  logger.debug('Wrote input array to img file')

  return new_file

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def check_envi_byte_order(f):
  """Check (and correct) byte order of img file (ENVI format)

  Parameters
  ----------
  f : img file (ENVI format)

  Returns
  -------
  new_file : return True/False if new img file has been created
  """

  # convert to pathlib object
  f = pathlib.Path(f).resolve()

  # check that file exists
  if not f.is_file():
    logger.error(f'Cannot find file: {f}')
    new_file = False
    return new_file

  # get envi header file
  hdr_file = f.parent / f'{f.name.split(".img")[0]}.hdr'

  logger.debug(f'hdr_file: {hdr_file}')

  # check that envi header file exists
  if not hdr_file.is_file():
    logger.error(f'Cannot find hdr_file: {hdr_file}')
    new_file = False
    return new_file

  # read lines from envi header file
  with open(hdr_file) as ff:
    header_contents = ff.read().splitlines()

  # get img_byte_order from header
  for header_line in header_contents:
    if 'byte order' in header_line:
      logger.debug(header_line)
      img_byte_order = int(header_line[-1])

  # get image data type
  img_data_type = gdal.Open(
    f.as_posix(), gdal.GA_ReadOnly
  ).GetRasterBand(1).DataType

  # get system byte order
  system_byte_order = sys.byteorder

  logger.info(f'img_data_type: {img_data_type}')
  logger.info(f'img_byte_order: {img_byte_order}')
  logger.info(f'sytem_byte_order: {system_byte_order}')

  # convert system_byte_order to integer
  if system_byte_order == 'little':
    system_byte_order = 0
  elif system_byte_order == 'big':
    system_byte_order = 1
  else:
    logger.error(f'Unknown system_byte_order')
    new_file = False
    return new_file

  # check if img and system byte orders match, re-write if needed
  if img_byte_order == system_byte_order:
    logger.info(f'system_byte_order and img_byte_order match')
    new_file = False
    return new_file
  else:
    logger.info(f'system_byte_order and img_byte_order do not match')
    logger.info(f'reading and re-writing image')
    img_data = get_all_bands(f.as_posix())
    new_file = write_img(
      f.as_posix(), img_data, as_type=img_data_type, overwrite=True
    )

  return new_file
  
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #


def embed_color_table_in_tif(tif_path, color_list):
  """Embed class color table to tif file

  Parameters
  ----------
  tif_path : path to input (and output) tif file
  color_list : list of colors to embed
  """

  logger.debug(f'Embedding color table in input tif image: {tif_path}')

  tmpDS = gdal.Open(tif_path, gdal.GA_Update)
  hBand = tmpDS.GetRasterBand(1)
  hCT = gdal.ColorTable()

  for inx in range(len(color_list)):
    hCT.SetColorEntry(
      inx, (
        color_list[inx][0],
        color_list[inx][1],
        color_list[inx][2],
        255
      )
    )

  hBand.SetRasterColorTable(hCT)

  hBand.FlushCache()
  tmpDS.FlushCache()

  hCT   = None
  hBand = None
  tmpDS = None

  return

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <read_write_img.py> ----
