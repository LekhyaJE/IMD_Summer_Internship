# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:54:18 2023

@author: lenovo
"""
##This script was made to analyse using daily accumulated precipitation from 1 single nc file(temporal resolution 1day)
#importing libraries
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
#from mpl_toolkits.basemap import Basemap
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
strurl=r"C:\Users\lenovo\Desktop\SIP_IMD_2023\GPM_data\Accum_daily\3B-DAY.MS.MRG.3IMERG.20150516-S000000-E235959.V06.nc4.SUB.nc4"
df=xr.open_dataset(strurl)
Date=str(strurl)
date=Date[79:87]


#print(df)
#print(df.variables.keys())
#print(df['T2'])
prcp=df['precipitationCal'].values.astype(float)
lats=df['lat'].values.astype(float)
long=df['lon'].values.astype(float)
PRCP=np.squeeze(prcp,axis=None)
X,Y=np.meshgrid(long,lats)
trans_PRCP = np.transpose(PRCP)

# print(X.shape)
print(trans_PRCP.shape)
    
# ####################################################################
fig, axarr = plt.subplots(1,1,dpi=300,facecolor='w', edgecolor='k',subplot_kw={'projection':
                                          ccrs.PlateCarree()})
fig.tight_layout()
axlist = np.ravel(axarr)#does the job of flattening array
ax=axlist[0]
levels = [10,30,70,90,100,110]
plot = ax.contourf(X,Y,trans_PRCP, cmap='rainbow',levels=levels, transform=ccrs.PlateCarree())   
cbar=plt.colorbar(plot,
#vmin=vmin, vmax=vmax,
ax=ax,
drawedges=False,

orientation='vertical',
# shrink=0.8,
pad=0.075,                    #extendfrac='auto',
extendrect=True)
cbar.ax.tick_params(labelsize=10)

##########################################
for ax in axlist:
    ax.set_title('Kol_obs'+str(date), loc='center',fontsize=10)
    ax.coastlines(resolution="50m",linewidth=1)
    ax.set_xticks([87.2, 87.7, 88.2,88.7])  # Example: longitude ticks
    ax.set_yticks([21.75,22.25, 22.75,23.25])  # Example: latitude ticks
    ax.set_extent([87.197258,89.13791656,21.70922852,23.50069427], crs=ccrs.PlateCarree())#Kolkata
    #ax.set_extent([85.172, 86.470,19.742,20.959], crs=ccrs.PlateCarree())# BBSR
    
    ax.coastlines(resolution="50m", linewidth=1)
    #ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='k', facecolor='grey'))

#Plot other data or features on the GeoAxes
...

#Display the plot
plt.show()

rainamt=np.sum(PRCP)
print(trans_PRCP.shape)
rainamt=rainamt/182
print('accumrainfall:',rainamt)

