import pathlib
from skimage.transform import PiecewiseAffineTransform, warp

try:
    from osgeo import gdal
except:
    import gdal
import numpy as np
import os
import csv
import sys
import glob
import xml.etree.ElementTree
sys.path.append(r'../geolocation_grid')
from LocationMapping import LocationMapping
import time