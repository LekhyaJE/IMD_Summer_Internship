# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:23:44 2023

@author: lenovo
"""
##This script was made to analyse using half hourly precipitation from multiple files downloaded using wget
#half hourly temporal resolution
import pandas as pd
import numpy as np
import xarray as xr
import requests 
import os
import glob
import netCDF4 as nc
from datetime import datetime, time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker



# folder_path = r"C:\Users\lenovo\Desktop\SIP_IMD_2023\GPM_data\Kolkata\2016050700"
# file_pattern = '*.nc4'

# Grabbing data from the required path
directory = r"C:\Users\lenovo\Desktop\SIP_IMD_2023\GPM_data\Kolkata\2015051600"



data = glob.glob(os.path.join(directory, '*.nc4'))
Date=str(directory)
date=Date[59:69]

#print(date)

#Extracting latitude and longitude from the first file
file = data[0]
ds = xr.open_dataset(file)
lons = ds['lon'].values.astype(float)
lats = ds['lat'].values.astype(float)
#time = ds['time'][:,0,0]
    # Meshgriding latitude and longitude
X, Y = np.meshgrid(lons, lats)

# Code for Rainfall data
array_list = []
Time_list=[]
for i in range(48):
    fname = data[i]
    ds = xr.open_dataset(fname)
    rain = ds['precipitationCal'].values.astype(float)
    time=ds['time'].values.astype(object)
    Time_list.append(time)
    df_rain = np.array(rain)
    array_list.append(df_rain)
    array_list_reshaped=np.squeeze(array_list)
rain_rate=[]
for i in range(48):   
    rainu=array_list_reshaped[i,:,:]
    rainsum=np.sum(rainu)
    rainavg=rainsum/961
    rain_rate.append(rainavg)
    
grouped_data = [sum(rain_rate[i:i+2]) for i in range(0, len(rain_rate), 2)]
new_time= [sum(Time_list[i:i+2]) for i in range(0, len(Time_list), 2)]
print(grouped_data)  
#print(len(array_list))
######################################################################################
# Plotting the data
# fig, ax = plt.subplots(1, 1, dpi=300, facecolor='w', edgecolor='k',
#                         subplot_kw={'projection': ccrs.PlateCarree()})
# fig.tight_layout()
# levels = [10, 20, 30, 40,50, 60,70,80,90,100,110,120,130,140,150,160,170,180,200]
# plot = ax.contourf(X, Y, array_list, cmap='tab20c',levels=levels, transform=ccrs.PlateCarree())

# cbar = plt.colorbar(plot, ax=ax, drawedges=False, orientation='vertical', shrink=0.8, pad=0.075,
#                     extendrect=True)
# cbar.ax.tick_params(labelsize=10)
# cbar.set_label('mm')
# # Set latitude and longitude ticks
# #ax.set_xticks([85.2, 85.7, 86.2,86.7])  # Example: longitude ticks BBSR
# #ax.set_yticks([19.75, 20.25, 20.75,21.25])  # Example: latitude ticks BBSR
# ax.set_title('kol'+str(date), loc='center', fontsize=10)
# #ax.set_extent([85.17223358154297,  86.47032928466797, 19.742523193359375, 20.959556579589844], crs=ccrs.PlateCarree())

# ax.coastlines(resolution="50m", linewidth=1)
#ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='k', facecolor='grey'))


############rainrate plot#######################
plt.plot(new_time, grouped_data, linestyle='-', color='b', label='Line Plot')
plt.show()

