# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:44:12 2023

@author: lenovo
"""

import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

strurl=r"C:\Users\lenovo\Desktop\no_urbfrac\KOL\rainfall\2015042100_ts_kol_d03_acc_rain.nc"
df=xr.open_dataset(strurl)

# Date=str(strurl)
# date=Date[45:55]
# print(date)
# accumulated_rainfall = np.sum(df[:, :, 1])
# print(df.variables.keys())
lats=df.variables['XLAT'][:,0]
longs=df.variables['XLONG'][0,:]

#BBSR Hotspot
# lats_hs=lats[60:110] #hotspot area
# long_hs=longs[76:126]

#Kolkata Hotspot
#lats_hs=lats[132:182]
#long_hs=longs[169:219]

X,Y = np.meshgrid(longs, lats)

list_lats = lats.values.tolist()
list_longs = longs.values.tolist()
#print(list_longs)
############## PLOTTING ##############################################
for i in range(97):
    df_rain=df.variables['rain'][i,:,:] #this is a 2d array
#The rainfall u getting is cumulative. So, the plot at the last time step is the accumulated rainfall plot
#and the sum of elements in the last 2d array is ur accum rainfall    
####################################################################
    fig, axarr = plt.subplots(1,1,dpi=300,facecolor='w', edgecolor='k',subplot_kw={'projection':
                                          ccrs.PlateCarree()})
    fig.tight_layout()
    axlist = np.ravel(axarr)#does the job of flattening array
    levels = [10, 20, 30, 40,50, 60,70,80,90,100,110,120,130,140,150,160,170,180,200]
    ax=axlist[0]
    plot = ax.contourf(X,Y,df_rain,cmap='tab20c',levels=levels, transform=ccrs.PlateCarree())
    cbar=plt.colorbar(plot,
                          ax=ax,
                          drawedges=False,
                          # ticks=np.arange(0,100,10),
                          orientation='vertical',
                          shrink=1,
                          pad=0.075,                    #extendfrac='auto',
                          extendrect=True)
    cbar.set_label('mm')
    cbar.ax.tick_params(labelsize=10)  
# ###########################################
    for ax in axlist:
        ax.set_extent([75, 100, 0, 30], crs=ccrs.PlateCarree())
        ax.coastlines(resolution="50m",linewidth=1)
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='grey'))
        
        
        plt.show()
        
#             ax.set_title('bbsr_nourban'+str(date),fontsize=10)
#             ax.coastlines(resolution="50m",linewidth=1)
            #BBSR Coords
#             start_value = 19.742523193359375 + 0.15
#             end_value = 20.959556579589844
#             num_ticks = 4  # Including the start and end values
#             ytick_values = np.linspace(start_value, end_value, num_ticks)

#             start_value = 85.17223358154297
#             end_value = 86.47032928466797
#             xtick_values = np.linspace(start_value, end_value, num_ticks)
#              ax.set_yticks(ytick_values)
#             ax.set_xticks(xtick_values)
#             # ax.set_yticklabels(list_lats)
#             # ax.set_xticklabels(list_longs)
#             plt.show()
    
#Find the accumulated rainfall
# accurain=df.variables['rain'][96,:,:]
# print(accurain.shape)
# sum_of_array=np.sum(accurain)
# Accum_rain=sum_of_array/40401
# print('Acuumulated rainfall: ',Accum_rain)
            