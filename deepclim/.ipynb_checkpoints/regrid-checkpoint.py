# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Regridder from curvilinear grid to retilinear grid
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.05.21
Last Update     : 2019.05.21
Contributor     :
Description     : This module provides several methods to perform regridding operations from curvilinear grid (ocean grid) 
                  to retilinear grid (atmosphere grid)
Library         : iris
Return Values   : np.array
Caveat!         :
"""

import numpy as np
import configargparse
import os
import iris
import iris.plot as iplt


class curv2rect:
    def __init__(self, field, gphi, glam):
        """
        Initialize the regridder module.
        We use the library iris to perform regridding from curvilinear grid to retilinear grid.
        Following the documentation of iris, all the methods are applied based on objects called 'cube'.
        more detials about those method:
        https://scitools.org.uk/iris/docs/latest/iris/iris/experimental/regrid.html
        param field: input array
        param gphi : latitude of curvilinear grid
        param glam : longitude of curvilinear grid
        """
        self.field = field
        self.gphi = gphi
        self.glam = glam
        # basic dimensions for cube in iris
        lat_iris = iris.coords.AuxCoord(self.gphi, standard_name='latitude', long_name='latitude',
                                    var_name='lat', units='degrees')
        lon_iris = iris.coords.AuxCoord(self.glam, standard_name='longitude', long_name='longitude',
                                    var_name='lon', units='degrees')
        # assembly the cube
        cube_iris = iris.cube.Cube(self.field, long_name='curvilinear field', var_name='field', 
                                   units='1', aux_coords_and_dims=[(lat_iris, (0,1)), (lon_iris, (0,1))])
        # choose the coordinate system
        coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        cube_iris.coord('latitude').coord_system = coord_sys
        cube_iris.coord('longitude').coord_system = coord_sys
        
        self.cube = cube_iris
        
    def visualization(self, nx=720, ny=240):
        """
        This module performs nearest neighbour interpolation for a quick visualization only.
        param nx: longitudinal slice
        param ny: latitudinal slice
        """
        # choose the projection map type
        projection = ccrs.PlateCarree()
        # Transform cube to target projection
        # this method only means for a fast visualization. We can not choose target coordinate
        cube_regrid, extent = iris.analysis.cartography.project(self.cube, projection,nx, ny)
        
        return cube_regrid
    
    def neighbour(self, lat_aux, lon_aux):
        """
        This module performs nearest neighbour interpolation for a given curvilinear grid to a retilinear grid.
        param lat_aux: laititude of the target grid (should be 1D)
        param lon_aux: longtitude of the target grid (should be 1D)
        """
        if lat_aux.shape != 1:
            msg = "The given latitude is not a one dimensional vector."
            raise configargparse.ArgumentTypeError(msg)
            
        if lon_aux.shape != 1:
            msg = "The given longitude is not a one dimensional vector."
            raise configargparse.ArgumentTypeError(msg)
        # assemble target cube
        lat_tar = iris.coords.DimCoord(lat_aux, standard_name='latitude',
                                       units='degrees_north', coord_system='GeogCS')
        lon_tar = iris.coords.DimCoord(lon_aux, standard_name='longitude',
                                       units='degrees_east', coord_system='GeogCS')
        dummy_data = np.zeros((len(lat_aux), len(lon_aux)))
        cube_tar = iris.cube.Cube(dummy_data,dim_coords_and_dims=[(lat_tar, 0), (lon_tar, 1)])
        # create the coordinate system for the target cube
        cube_tar.coord('latitude').guess_bounds()
        cube_tar.coord('longitude').guess_bounds()
        cube_tar.coord('latitude').coord_system = coord_sys
        cube_tar.coord('longitude').coord_system = coord_sys
        # create a weight matrix for regridding
        weights = np.ones(self.cube.shape)
        # get regridder from given cubes
        base = iris.analysis.UnstructuredNearest()
        regridder = base.regridder(self.cube,cube_tar)
        # Transform cube to target projection
        cube_regrid = regridder(self.cube)
        
        return cube_regrid
    
    def weight(self, lat_aux, lon_aux, weights_matrix):
         """
        This module performs interpolation for a given curvilinear grid to a retilinear grid with given weight matrix.
        param lat_aux: laititude of the target grid (should be 1D)
        param lon_aux: longtitude of the target grid (should be 1D)
        param weights_matrix: Weights for the regrdding
        """
        if lat_aux.shape != 1:
            msg = "The given latitude is not a one dimensional vector."
            raise configargparse.ArgumentTypeError(msg)
            
        if lon_aux.shape != 1:
            msg = "The given longitude is not a one dimensional vector."
            raise configargparse.ArgumentTypeError(msg)
            
        if self.cube.shape != weights_matrix.shape:
            msg = "The given weight matrix should have the same shape as the input field."
            raise configargparse.ArgumentTypeError(msg)
            
        # assemble target cube
        lat_tar = iris.coords.DimCoord(lat_aux, standard_name='latitude',
                                       units='degrees_north', coord_system='GeogCS')
        lon_tar = iris.coords.DimCoord(lon_aux, standard_name='longitude',
                                       units='degrees_east', coord_system='GeogCS')
        dummy_data = np.zeros((len(lat_aux), len(lon_aux)))
        cube_tar = iris.cube.Cube(dummy_data,dim_coords_and_dims=[(lat_tar, 0), (lon_tar, 1)])
        # create the coordinate system for the target cube
        cube_tar.coord('latitude').guess_bounds()
        cube_tar.coord('longitude').guess_bounds()
        cube_tar.coord('latitude').coord_system = coord_sys
        cube_tar.coord('longitude').coord_system = coord_sys
        # create a weight matrix for regridding
        weights = weights_matrix
        #
        cube_regrid = iris.experimental.regrid.regrid_weighted_curvilinear_to_rectilinear(self.cube,weights,cube_tar)
        
        return cube_regrid