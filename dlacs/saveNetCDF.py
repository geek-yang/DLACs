# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Save Output Files into NetCDF files
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2020.03.11
Last Update     : 2020.04.10
Contributor     :
Description     : This module aims to save the output fields into the netCDF files.
                  It outputs netCDF 4 files. The data has been compressed.

Return Values   : netCDF files
"""

import sys
import os
import numpy as np
from netCDF4 import Dataset

class savenc:
    def __init__(self, datapath, filename):
        """
        Save the output fields into netCDF files.
        """
        print ("Save output fields as netCDF4 files.")
        self.datapath = datapath
        # check if file exists
        if os.path.isfile(os.path.join(datapath, filename)):
            print ("File already exist.")
            raise ValueError('Should use a different name.')
        else:
            self.fileName = filename
        
    def ncfile(self, numpyMatrix):
        """
        Create nc4 file with given dataset. The dataset should contain 4 dimensions.
        """
        data_wrap = Dataset(os.path.join(self.datapath,self.fileName), 'w',format = 'NETCDF4')
        # get the dimension of input data
        week, lead, lat, lon = numpyMatrix.shape
        # create dimensions for netcdf data
        week_wrap_dim = data_wrap.createDimension('week',week)
        lead_wrap_dim = data_wrap.createDimension('lead',lead)
        lat_wrap_dim = data_wrap.createDimension('latitude',lat)
        lon_wrap_dim = data_wrap.createDimension('longitude',lon)
        # create 1-dimension variables
        week_wrap_var = data_wrap.createVariable('year',np.int32,('week',))
        lead_wrap_var = data_wrap.createVariable('month',np.int32,('lead',))
        lat_wrap_var = data_wrap.createVariable('latitude',np.float32,('latitude',))
        lon_wrap_var = data_wrap.createVariable('longitude',np.float32,('longitude',))
        # create 4-dimension variables
        pred_wrap_var = data_wrap.createVariable('pred',np.float32,('week','lead','latitude','longitude'))
        # global attributes
        data_wrap.description = 'Lead time dependent forecast field'
        # variable attributes
        lat_wrap_var.units = 'degree_north'
        lon_wrap_var.units = 'degree_east'

        week_wrap_var.long_name = 'forecast week'
        lead_wrap_var.long_name = 'lead time'
        pred_wrap_var.long_name = 'predictand'
        # writing data
        week_wrap_var[:] = week
        lead_wrap_var[:] = lead
        lat_wrap_var[:] = lat
        lon_wrap_var[:] = lon
        pred_wrap_var[:] = numpyMatrix
        # close the file
        data_wrap.close()
        print("Create netcdf files successfully!!")

    def ncfile_Lorenz(self, numpyMatrix):
        """
        Create nc4 file with given dataset. The dataset should contain 4 dimensions.
        This module is specifically designed for Lorenz 84 model output.
        """
        data_wrap = Dataset(os.path.join(self.datapath,self.fileName), 'w',format = 'NETCDF4')
        # get the dimension of input data
        var, time, lead = numpyMatrix.shape
        # create dimensions for netcdf data
        var_wrap_dim = data_wrap.createDimension('var',var)
        time_wrap_dim = data_wrap.createDimension('time',time)
        lead_wrap_dim = data_wrap.createDimension('lead',lead)
        # create 1-dimension variables
        var_wrap_var = data_wrap.createVariable('var',np.int32,('var',))
        time_wrap_var = data_wrap.createVariable('time',np.int32,('time',))
        lead_wrap_var = data_wrap.createVariable('lead',np.int32,('lead',))
        # create 3-dimension variables
        series_wrap_var = data_wrap.createVariable('series',np.float32,('var', 'time', 'lead'))
        # global attributes
        data_wrap.description = 'Lead time dependent Lorenz 84 forecast'
        # variable attributes
        var_wrap_var.units = '1'
        time_wrap_var.units = '1'
        lead_wrap_var.units = '1'

        var_wrap_var.long_name = 'number of variables'
        time_wrap_var.long_name = 'time span'
        lead_wrap_var.long_name = 'lead time'
        # writing data
        var_wrap_var[:] = var
        time_wrap_var[:] = time
        lead_wrap_var[:] = lead
        series_wrap_var[:] = numpyMatrix
        # close the file
        data_wrap.close()
        print("Create netcdf files successfully!!")