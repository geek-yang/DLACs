# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Plots generator for visualization
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2018.08.13
Last Update     : 2018.08.13
Contributor     :
Description     : This module provides several methods to perform statistical
                  analysis on MET and all kinds of fields.
Return Values   : pngs
Caveat!         : The style of gridliner of Cartopy can be found at
                  https://scitools.org.uk/cartopy/docs/v0.13/matplotlib/gridliner.html
"""

import numpy as np
import scipy
#from scipy import stats
import os
import matplotlib
#import seaborn as sns
#import bokeh
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import iris
import iris.plot as iplt
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

class plots:
    @staticmethod
    def linearRegress(xaxis, corr, figname='./LinearRegression.png'):
        """
        This module will make a x-y plot to display the correlation coefficient
        got from the linear regression.
        
        param xaxis: latitude for the plot as x axis
        param corr: the correlation coefficient
        param figname: name and output path of figure
        return: Figures
        rtype: png
        """
        print ("Create x-y plot of correlation coefficient.")
        fig = plt.figure()
        plt.plot(xaxis, corr)
        plt.xlabel("Latitude")
        #plt.xticks(np.linspace(20, 90, 11))
        plt.ylabel("Correlation Coefficient")
        plt.show()
        fig.savefig(figname,dpi=400)
        plt.close(fig)
        
    @staticmethod
    def vertProfile(xaxis, yaxis, corr, p_value, label,
                    ticks, figname='./VerticalProfile', ttest=False):
        """
        This module helps to create a plot to show the vertical profile of fields
        after regression.
        
        param xaxis: latitude for the plot as x axis
        param yaxis: level for the plot as y axis
        param corr: the correlation coefficient
        param figname: name and output path of figure
        return: Figures
        rtype: png        
        """
        print ("Create contour plot of correlation coefficient for vertical profiles.")
        # make plots
        fig = plt.figure(figsize=(6.5,5.4))        
        cs = plt.contourf(xaxis, yaxis, corr, levels=ticks, cmap='coolwarm', extend='both')
        cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                            shrink =0.8, pad=0.135, format="%.1f")
        cbar.set_label(label,size = 10)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize = 10)
        if ttest == True:
            ii, jj = np.where(p_value<=0.05) # 95% significance
            plt.plot(xaxis[jj], yaxis[ii], 'go', alpha=0.3)
        plt.xlabel("Latitude")
        plt.ylabel("Level (hPa)")
        #invert the y axis
        plt.gca().invert_yaxis()
        plt.show()
        fig.savefig(figname,dpi=200)
        plt.close(fig)
        
    @staticmethod
    def vertProfileSig(xaxis, yaxis, corr, p_value, label,
                       ticks, figname='./VerticalProfile', ttest=False):
        """
        This module helps to create a plot to show the vertical profile of fields
        after regression. It also includes the full contour of confidence interval.
        
        param xaxis: latitude for the plot as x axis
        param yaxis: level for the plot as y axis
        param corr: the correlation coefficient
        param figname: name and output path of figure
        return: Figures
        rtype: png        
        """
        print ("Create contour plot of correlation coefficient for vertical profiles.")
        # make plots
        contour_level = [i for i in np.arange(0,1.1, 0.1)]
        fig = plt.figure(figsize=(6.5,5.4))        
        cs = plt.contourf(xaxis, yaxis, corr, levels=ticks, cmap='coolwarm', extend='both')
        cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                            shrink =0.8, pad=0.135, format="%.1f")
        cbar.set_label(label,size = 10)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize = 10)
        plt.xlabel("Latitude")
        plt.ylabel("Level (hPa)")
        cs = plt.contour(xaxis, yaxis, 1-p_value,
                         contour_level, colors='k')
        plt.clabel(cs, inline=1, fontsize=10)
        #invert the y axis
        plt.gca().invert_yaxis()
        plt.show()
        fig.savefig(figname,dpi=200)
        plt.close(fig)
        
    @staticmethod
    def vertProfileOverlap(xaxis, yaxis, corr, cont, p_value, label,
                           ticks, contour_level, inline_space,
                           figname='./VerticalProfile', ttest=False):
        """
        This module helps to create a plot to show the vertical profile of fields
        after regression. It also includes the full contour of stokes stream function.
        
        param xaxis: latitude for the plot as x axis
        param yaxis: level for the plot as y axis
        param corr: the correlation coefficient
        param figname: name and output path of figure
        return: Figures
        rtype: png        
        """
        print ("Create contour plot of stokes stream function for vertical profiles.")
        fig = plt.figure(figsize=(6.5,5.4))        
        cs = plt.contourf(xaxis, yaxis, corr, levels=ticks, cmap='coolwarm', extend='both')
        cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                            shrink =0.8, pad=0.135, format="%.1f")
        cbar.set_label(label,size = 10)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize = 10)
        if ttest == True:
            ii, jj = np.where(p_value<=0.05) # 95% significance
            plt.plot(xaxis[jj], yaxis[ii], 'go', alpha=0.3)
        plt.xlabel("Latitude")
        plt.ylabel("Level (hPa)")
        contour = plt.contour(xaxis, yaxis, cont,
                              contour_level, colors='k', linewidths = 0.9, alpha=0.6)
        plt.clabel(contour, inline=inline_space, fontsize=8, fmt = '%1.1f')        
        #invert the y axis
        plt.gca().invert_yaxis()
        plt.show()
        fig.savefig(figname,dpi=200)
        plt.close(fig)
        
    @staticmethod    
    def leadlagRegress(yaxis, corr, lag, p_value, figname='./LeadLagRegression.png',
                       ttest=False):
        """
        This module will make a contour plot to display the correlation coefficient
        got from the lead/lag regression.
        
        param yaxis: latitude for the plot as y axis
        param corr: the correlation coefficient
        param lag: the maximum lag time for plot as x axis
        param figname: name and output path of figure
        return: Figures
        rtype: png
        """
        print ("Create contour plot of correlation coefficient.")
        # calculate the lead/lag index as x axis
        lag_index = np.arange(-lag,lag+1,1)
        xaxis = lag_index / 12
        # make plots
        fig = plt.figure()
        #contour_level = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
        contour_level = np.array([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
        cs = plt.contour(xaxis, yaxis, corr.transpose(),
                         contour_level, colors='k')
        plt.clabel(cs, inline=1, fontsize=10)
        if ttest == True:
            ii, jj = np.where(p_value.transpose()<=0.05) # 95% significance
            plt.scatter(xaxis[jj], yaxis[ii], s=0.8, c='gray', alpha=0.6)
            #plt.plot(xaxis[jj], yaxis[ii], 'go', s=0.1, alpha=0.3)
        plt.xlabel("Time Lag (year)")
        #lead_year = ['-15','-12','-9','-6','-3','0','3','6','9','12','15']
        plt.ylabel("Latitude")
        plt.show()
        fig.savefig(figname,dpi=200)
        plt.close(fig)
    
    @staticmethod
    def geograph(latitude, longitude, field, label, ticks,
                 figname='./NorthPolar.png', gridtype='geographical',
                 boundary='northhem', colormap= 'coolwarm'):
        """
        This module will make a geographical plot to give a spatial view of fields.
        This module is built on iris and cartopy for the visualization of fields on
        both geographical and curvilinear grid.
        
        param lat: latitude coordinate for plot
        param lon: longitude coordinate for plot
        param field: input field for visualization
        param gridtype: type of input spatial fields, it has two options
        - geographical (default) the coordinate is geographical, normally applied to atmosphere reanalysis
        - curvilinear the coordinate is curvilinear, normally applied to ocean reanalysis
        param figname: name and output path of figure
        param boundary: region for plot. It determines the boundary of plot area (lat,lon) and projection.
        - northhem (default) plot the north hemisphere from 20N-90N & 180W-180E, with the projection NorthPolarStereo.
        - atlantic plot the north Atlantic from 20N-90N & 90W-40E, with the projection PlateCarree
        return: figures
        rtype: png
        """
        print ("Create a NorthPolarStereo view of input fields.")
        if gridtype == 'geographical':
            print ("The input fields are originally on geographical grid")
            # first construct iris coordinate
            lat_iris = iris.coords.DimCoord(latitude, standard_name='latitude', long_name='latitude',
                                            var_name='lat', units='degrees')
            lon_iris = iris.coords.DimCoord(longitude, standard_name='longitude', long_name='longitude',
                                            var_name='lon', units='degrees')
            # assembly the cube
            cube_iris = iris.cube.Cube(field, long_name='geographical field', var_name='field', 
                                       units='1', dim_coords_and_dims=[(lat_iris, 0), (lon_iris, 1)])
            if boundary == 'northhem':
                fig = plt.figure()
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                #ax.set_extent([-180,180,20,90],ccrs.PlateCarree())
                ax.set_extent([-180,180,50,90],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--')
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
                cs = iplt.contourf(cube_iris, cmap=colormap,levels=ticks, extend='both') #, vmin=ticks[0], vmax=ticks[-1]
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 8)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 6)
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
            elif boundary == 'atlantic':
                fig = plt.figure(figsize=(8,5.4))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([-90,40,20,85],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlabels_top = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': 11, 'color': 'gray'}
                gl.ylabel_style = {'size': 11, 'color': 'gray'}
                cs = iplt.contourf(cube_iris,cmap=colormap, levels=ticks, extend='both')
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 11)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 11)
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
            elif boundary == 'Barents_PlateCarree':
                fig = plt.figure(figsize=(6,5.4))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([15,65,60,85],ccrs.PlateCarree()) # W:18 E:60 S:64 N:80
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlabels_top = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': 11, 'color': 'gray'}
                gl.ylabel_style = {'size': 11, 'color': 'gray'}
                cs = iplt.contourf(cube_iris,cmap=colormap, levels=ticks, extend='both')
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 11)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 11)
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
            elif boundary == 'Barents_Polar':
                fig = plt.figure()
                ax = plt.axes(projection=ccrs.EquidistantConic(central_longitude=39.0, central_latitude=72.0))
                ax.set_extent([16,60,60,82],ccrs.PlateCarree()) # W:18 E:60 S:64 N:80
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--')
                cs = iplt.contourf(cube_iris, cmap=colormap,levels=ticks, extend='both', vmin=ticks[0], vmax=ticks[-1])
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 9)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 8)
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)            
            else:
                print ('This boundary is not supported by the module. Please check the documentation.')
        elif gridtype == 'curvilinear':
            print ("The input fields are originally on curvilinear grid")
            # first construct iris coordinate
            lat_iris = iris.coords.AuxCoord(latitude, standard_name='latitude', long_name='latitude',
                                            var_name='lat', units='degrees')
            lon_iris = iris.coords.AuxCoord(longitude, standard_name='longitude', long_name='longitude',
                                            var_name='lon', units='degrees')
            # assembly the cube
            cube_iris = iris.cube.Cube(field, long_name='curvilinear field', var_name='field', 
                                       units='1', aux_coords_and_dims=[(lat_iris, (0,1)), (lon_iris, (0,1))])
            coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
            cube_iris.coord('latitude').coord_system = coord_sys
            cube_iris.coord('longitude').coord_system = coord_sys
            # determine nx and ny for interpolation
            jj, ii = latitude.shape
            if ii > 1000:
                nx = 1440
                ny = 350
            else:
                nx = 720
                ny = 140
            cube_regrid, extent = iris.analysis.cartography.project(cube_iris, ccrs.PlateCarree(), nx, ny)
            # make plots
            if boundary == 'northhem':
                fig = plt.figure()
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-180,180,20,90],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--')
                #gl.ylocator = mticker.FixedLocator([50,60,70,80,90])
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
                cs = iplt.contourf(cube_regrid, cmap=colormap, vmin=ticks[0], vmax=ticks[-1]) #pcolormesh
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 8)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 6)
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
            elif boundary == 'atlantic':
                fig = plt.figure(figsize=(8,5.4))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([-90,40,20,85],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                #gl.ylocator = mticker.FixedLocator([50,60,70,80,90])
                gl.xlabels_top = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': 11, 'color': 'gray'}
                gl.ylabel_style = {'size': 11, 'color': 'gray'}
                cs = iplt.contourf(cube_regrid, cmap=colormap, vmin=ticks[0], vmax=ticks[-1])
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05, format="%.1f")
                cbar.set_label(label,size = 8)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 6)
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)                
            else:
                print ('This boundary is not supported by the module. Please check the documentation.')                
        else:
            raise IOError("This module only support fields on geographical or curvilinear grid!")
            
    @staticmethod
    def geograph_mode(latitude_x, longitude_x, field_x,
                      latitude_y, longitude_y, field_y, label, ticks, level,
                      figname='./NorthPolar.png', gridtype='geographical',
                      boundary='northhem', colormap= 'coolwarm'):
        """
        This module is designed for PCA/SVD/MCA analysis to illustrate two
        fields at the same time.
        This module is built on iris and cartopy for the visualization of fields on
        both geographical and curvilinear grid.
        
        param latitude_x: latitude coordinate for input field x
        param longitude_x: longitude coordinate for input field x
        param field_x: input field for visualization with shades
        param latitude_y: latitude coordinate for input field y
        param longitude_y: longitude coordinate for input field y
        param field_y: input field for visualization with contours
        param label: label of shades
        param ticks: ticks of shades
        param level: level of contour lines
        param gridtype: type of input spatial fields, it has two options
        - geographical (default) the coordinate is geographical, normally applied to atmosphere reanalysis
        - curvilinear the coordinate is curvilinear, normally applied to ocean reanalysis
        param figname: name and output path of figure
        param boundary: region for plot. It determines the boundary of plot area (lat,lon) and projection.
        - northhem (default) plot the north hemisphere from 20N-90N & 180W-180E, with the projection NorthPolarStereo.
        - atlantic plot the north Atlantic from 20N-90N & 90W-40E, with the projection PlateCarree
        return: figures
        rtype: png
        """
        print ("Create a NorthPolarStereo view of input fields.")
        if gridtype == 'geographical':
            print ("The input fields are originally on geographical grid")
            # mode variable x
            # first construct iris coordinate
            lat_iris_x = iris.coords.DimCoord(latitude_x, standard_name='latitude', long_name='latitude',
                                              var_name='lat', units='degrees')
            lon_iris_x = iris.coords.DimCoord(longitude_x, standard_name='longitude', long_name='longitude',
                                              var_name='lon', units='degrees')
            # assembly the cube
            cube_iris_x = iris.cube.Cube(field_x, long_name='geographical field', var_name='field', 
                                         units='1', dim_coords_and_dims=[(lat_iris_x, 0), (lon_iris_x, 1)])
            # mode variable y
            lat_iris_y = iris.coords.DimCoord(latitude_y, standard_name='latitude', long_name='latitude',
                                              var_name='lat', units='degrees')
            lon_iris_y = iris.coords.DimCoord(longitude_y, standard_name='longitude', long_name='longitude',
                                              var_name='lon', units='degrees')            
            cube_iris_y = iris.cube.Cube(field_y, long_name='geographical field', var_name='field', 
                                         units='1', dim_coords_and_dims=[(lat_iris_y, 0), (lon_iris_y, 1)])
            if boundary == 'northhem':
                fig = plt.figure()
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                #ax.set_extent([-180,180,20,90],ccrs.PlateCarree())
                ax.set_extent([-180,180,50,90],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--')
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
                cs = iplt.contourf(cube_iris_x, cmap=colormap,levels=ticks, extend='both') #, vmin=ticks[0], vmax=ticks[-1]
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 8)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 6)
                contour = iplt.contour(cube_iris_y, colors='dimgrey', levels = level,  linewidths = 0.8, format="%.2f")
                plt.clabel(contour, inline=True, fontsize = 8, fmt ="%.2f")
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
            elif boundary == 'atlantic':
                fig = plt.figure(figsize=(8,5.4))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([-90,40,20,85],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlabels_top = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': 11, 'color': 'gray'}
                gl.ylabel_style = {'size': 11, 'color': 'gray'}
                cs = iplt.contourf(cube_iris_x,cmap=colormap, levels=ticks, extend='both')
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 11)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 11)
                contour = iplt.contour(cube_iris_y, colors='dimgrey', levels = level,  linewidths = 0.8, format="%.2f")
                plt.clabel(contour, inline=True, fontsize = 8, fmt ="%.2f")
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
            elif boundary == 'barents_plateCarree':
                fig = plt.figure(figsize=(6,5.4))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([15,65,60,85],ccrs.PlateCarree()) # W:18 E:60 S:64 N:80
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlabels_top = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': 11, 'color': 'gray'}
                gl.ylabel_style = {'size': 11, 'color': 'gray'}
                cs = iplt.contourf(cube_iris_x,cmap=colormap, levels=ticks, extend='both')
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 11)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 11)
                contour = iplt.contour(cube_iris_y, colors='dimgrey', levels = level,  linewidths = 0.8, format="%.2f")
                plt.clabel(contour, inline=True, fontsize = 8, fmt ="%.2f")
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
            elif boundary == 'barents_polar':
                fig = plt.figure()
                ax = plt.axes(projection=ccrs.EquidistantConic(central_longitude=39.0, central_latitude=72.0))
                ax.set_extent([16,60,60,82],ccrs.PlateCarree()) # W:18 E:60 S:64 N:80
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--')
                cs = iplt.contourf(cube_iris_x, cmap=colormap,levels=ticks, extend='both', vmin=ticks[0], vmax=ticks[-1])
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 9)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 8)
                contour = iplt.contour(cube_iris_y, colors='dimgrey', levels = level,  linewidths = 0.8)
                plt.clabel(contour, inline=True, fontsize = 8, fmt ="%.2f")
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)            
            else:
                print ('This boundary is not supported by the module. Please check the documentation.')
        elif gridtype == 'curvilinear':
            print ("The input fields are originally on curvilinear grid")
            # mode variable x
            # first construct iris coordinate
            lat_iris_x = iris.coords.AuxCoord(latitude_x, standard_name='latitude', long_name='latitude',
                                            var_name='lat', units='degrees')
            lon_iris_x = iris.coords.AuxCoord(longitude_x, standard_name='longitude', long_name='longitude',
                                            var_name='lon', units='degrees')
            # assembly the cube
            cube_iris_x = iris.cube.Cube(field_x, long_name='curvilinear field', var_name='field', 
                                         units='1', aux_coords_and_dims=[(lat_iris_x, (0,1)), (lon_iris_x, (0,1))])
            coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
            cube_iris_x.coord('latitude').coord_system = coord_sys
            cube_iris_x.coord('longitude').coord_system = coord_sys
            # mode variable y
            lat_iris_y = iris.coords.AuxCoord(latitude_y, standard_name='latitude', long_name='latitude',
                                              var_name='lat', units='degrees')
            lon_iris_y = iris.coords.AuxCoord(longitude_y, standard_name='longitude', long_name='longitude',
                                              var_name='lon', units='degrees')
            cube_iris_y = iris.cube.Cube(field_y, long_name='curvilinear field', var_name='field', 
                                         units='1', aux_coords_and_dims=[(lat_iris_y, (0,1)), (lon_iris_y, (0,1))])
            cube_iris_y.coord('latitude').coord_system = coord_sys
            cube_iris_y.coord('longitude').coord_system = coord_sys
            # determine nx and ny for interpolation
            jj, ii = latitude.shape
            if ii > 1000:
                nx = 1440
                ny = 350
            else:
                nx = 720
                ny = 140
            cube_regrid_x, extent = iris.analysis.cartography.project(cube_iris_x, ccrs.PlateCarree(), nx, ny)
            cube_regrid_y, extent = iris.analysis.cartography.project(cube_iris_y, ccrs.PlateCarree(), nx, ny)
            # make plots
            if boundary == 'northhem':
                fig = plt.figure()
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-180,180,20,90],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--')
                #gl.ylocator = mticker.FixedLocator([50,60,70,80,90])
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
                cs = iplt.contourf(cube_regrid_x, cmap=colormap, vmin=ticks[0], vmax=ticks[-1]) #pcolormesh
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05)#, format="%.1f")
                cbar.set_label(label,size = 8)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 6)
                contour = iplt.contour(cube_regrid_y, colors='dimgrey', levels = level,  linewidths = 0.8)
                plt.clabel(contour, inline=True, fontsize = 8, fmt ="%.2f")
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)
            elif boundary == 'atlantic':
                fig = plt.figure(figsize=(8,5.4))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([-90,40,20,85],ccrs.PlateCarree())
                ax.set_aspect('1')
                ax.coastlines()
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                #gl.ylocator = mticker.FixedLocator([50,60,70,80,90])
                gl.xlabels_top = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': 11, 'color': 'gray'}
                gl.ylabel_style = {'size': 11, 'color': 'gray'}
                cs = iplt.contourf(cube_regrid_x, cmap=colormap, vmin=ticks[0], vmax=ticks[-1])
                cbar = fig.colorbar(cs,extend='both', orientation='horizontal',
                                    shrink =0.8, pad=0.05, format="%.1f")
                cbar.set_label(label,size = 8)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize = 6)
                contour = iplt.contour(cube_regrid_y, colors='dimgrey', levels = level,  linewidths = 0.8)
                plt.clabel(contour, inline=True, fontsize = 8, fmt ="%.2f")
                iplt.show()
                fig.savefig(figname, dpi=200)
                plt.close(fig)                
            else:
                print ('This boundary is not supported by the module. Please check the documentation.')                
        else:
            raise IOError("This module only support fields on geographical or curvilinear grid!")
        
def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_kwargs=None, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)