
#!/usr/bin/env python3
"""
Class for mapping between lat-lon coordinates and image projection coordinate, and raster coordinates inside image.

2021-10-22 Anders Gunnar Felix Hildeman

"""

try:
    from osgeo import osr
except:
    import osr

import numpy as np


class LocationMapping:


    def __init__( self, geotransform, projection ):

        xoffset, px_w, py_w, yoffset, px_h, py_h = geotransform

        self.trans_matrix = np.array( [ [px_w, py_w], [px_h, py_h ] ] )
        self.trans_offset = np.array( [xoffset, yoffset] ).reshape( (2, 1) ) + 0.5 * np.matmul( self.trans_matrix, np.ones((2,1)) )

        # get CRS from dataset
        crs = osr.SpatialReference()
        crs.ImportFromWkt( projection )
        # create lat/long crs with WGS84 datum
        crsGeo = osr.SpatialReference()
        crsGeo.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        #crsGeo.ImportFromEPSG(4326) # 4326 is the EPSG id of lat/long crs
        self.t = osr.CoordinateTransformation(crs, crsGeo)
        self.t_inv = osr.CoordinateTransformation(crsGeo, crs)


    def mapFromCoords2Proj( self, lat, long ):
        '''
            Map from standard projection (4326) to image projection
        '''

        points = np.stack( (long, lat), axis = 1 )
        pos = self.t_inv.TransformPoints( points )
        y, x = ( np.array( [elem[0] for elem in pos] ), np.array( [elem[1] for elem in pos] ) ) 

        return x, y


    def mapFromProj2Coords( self, x, y ):
        '''
            Map from image projection to standard projection (4326)
        '''

        points = np.stack( (x, y), axis = 1 )
        points = self.t.TransformPoints( points )
        long, lat = ( np.array([elem[0] for elem in points]), np.array([ elem[1] for elem in points ]) )

        return (lat, long)


    def mapFromRaster2Proj( self, x, y ):
        '''
            Map from raster points to projection
        '''

        points = np.stack( (x,y), axis = 1 ).T
        points = np.matmul( self.trans_matrix, points )
        points = points + self.trans_offset.reshape( (2,1) )

        return (points[0, :], points[1, :])


    def mapFromProj2Raster( self, x, y ):
        '''
            Map from projection points to raster
        '''

        points = np.stack( (x,y), axis = 1 ).T
        points = points - self.trans_offset.reshape( (2,1) )
        points = np.linalg.solve( self.trans_matrix, points )

        return (points[0, :], points[1, :])


    def latLon2Raster( self, lat, long, export = False ):
        '''
            Map from lat long coordinates to raster points
        '''
        #print("Check if coordinates are transformed right. If numbers doesn't make sense, it is due to gdal version. Install a different version or go to ./tools/geolocation_grid/LocationMapping.py and swap x and y in latLon2Raster function in 'self.mapFromProj2Raster(y, x )'\nMin lat and long:",  np.nanmin(lat), np.nanmin(long) )
        x, y = self.mapFromCoords2Proj(lat,long)
        #print("Min x and y", np.nanmin(x),np.nanmin(y))
        x, y = self.mapFromProj2Raster(y, x ) #swaped x and y 
        #print("Min rows and columns", np.nanmin(x),np.nanmin(y))
        

        return (x, y)


    def raster2LatLon( self, x, y ):
        '''
            Map from raster points to lat long coordinates
        '''

        x, y = self.mapFromRaster2Proj( x, y )
        lon, lat = self.mapFromProj2Coords( x, y )

        return (lat, lon)
