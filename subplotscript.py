

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:55:53 2023

@author: lenovo
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Data for plotting


# UF Data
strurl=r"C:/Users/lenovo/Desktop/SIP_IMD_2023/uf_extracts_MYJ/kolkata/rainfall/2015042100_ts_kol_d03_acc_rain.nc"
df=xr.open_dataset(strurl)
uf_rain=df.variables['rain'][96,:,:]

# CTL Data
ctlurl=r"C:/Users/lenovo/Desktop/SIP_IMD_2023/no_urbfrac/KOL/rainfall/2015042100_ts_kol_d03_acc_rain.nc"
df_ctl=xr.open_dataset(ctlurl)
ctl_rain=df_ctl.variables['rain'][96,:,:]

# Meshgridding for Model data
lats=df.variables['XLAT'][:,0]
longs=df.variables['XLONG'][0,:]
X,Y=np.meshgrid(longs,lats)


# OBS Data
strurl_obs=r"C:/Users/lenovo/Desktop/SIP_IMD_2023/GPM_data/Accum_daily/3B-DAY.MS.MRG.3IMERG.20150421-S000000-E235959.V06.nc4.nc4"
df_obs=xr.open_dataset(strurl_obs)
Date=str(strurl_obs)
prcp=df_obs['precipitationCal'].values.astype(float)
olats=df_obs['lat'].values.astype(float)
olong=df_obs['lon'].values.astype(float)
PRCP=np.squeeze(prcp,axis=None)
XO,YO=np.meshgrid(olong,olats)
trans_PRCP = np.transpose(PRCP)


##################################################################################################
# Create a figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6),gridspec_kw={'wspace': 0.001}, edgecolor='k',subplot_kw={'projection':
                                    ccrs.PlateCarree()})
    
fig.tight_layout()
axlist = np.ravel(axs)#does the job of flattening array
levels = [10,30,50,70,90,110,130]


nan_prcp = np.where((trans_PRCP >= 0) & (trans_PRCP <= 9.9), np.nan, trans_PRCP)


# Plot each subplot and set border properties
##########################################################################################
cmap_reversed = plt.cm.Spectral.reversed()
cmap_colors = cmap_reversed(np.linspace(0, 1, cmap_reversed.N))
color_to_replace = 0 
cmap_colors[color_to_replace] = [1, 1, 1, 1]  # Replace the color with white
white_replaced_cmap = LinearSegmentedColormap.from_list('white_replaced', cmap_colors, cmap_reversed.N)
ax=axlist[0]
plot=ax.contourf(XO,YO,nan_prcp,cmap=white_replaced_cmap,levels=levels,extend='both', transform=ccrs.PlateCarree(),)
ax.set_title('GPM', loc='center',fontsize=20)
ax.set_yticks([22,22.5,23,23.5])#Kolkata
ax.set_xticks([87.5,88,88.5,89])#Kolkata
# ax.set_yticks([19.8,20.3,20.8,21.3])
# ax.set_xticks([85.2,85.7,86.2,86.7])


ax.spines['top'].set_linewidth(10)


nan_ctl_rain = np.where((ctl_rain >= 0) & (ctl_rain <= 9.9), np.nan, ctl_rain)
ax=axlist[1]
plot=ax.contourf(X,Y,nan_ctl_rain,cmap=white_replaced_cmap,levels=levels,extend='both', transform=ccrs.PlateCarree())
ax.set_title('CTL', loc='center',fontsize=20)
ax.set_xticks([87.5,88,88.5,89])
#ax.set_xticks([85.2,85.7,86.2,86.7])


nan_uf_rain = np.where((uf_rain >= 0) & (uf_rain <= 9.9), np.nan, uf_rain)
ax=axlist[2]
plot=ax.contourf(X,Y,nan_uf_rain,cmap=white_replaced_cmap,levels=levels,extend='both', transform=ccrs.PlateCarree())
ax.set_title('UF', loc='center',fontsize=20)
ax.set_xticks([87.5,88,88.5,89])
#ax.set_xticks([85.2,85.7,86.2,86.7])




################################################################################

#################################################################################
axbar=[axlist[0],axlist[1],axlist[2]]
cbar=plt.colorbar(plot,
                      ax=axbar,
                      #ticks=np.arange(-40, 40.1, 10),
                      drawedges=False,
                      orientation='horizontal',
                      shrink=0.4,
                      pad=0.15,
                      #extendfrac='auto'
                      )               
cbar.ax.tick_params(labelsize=20)

for ax in axlist:
    #ax.set_title('vor', loc='left',fontsize=12)
    #ax.set_title('$s^{-1}$', loc='right',fontsize=12)
    ax.set_extent([87.197258, 89.13791656, 21.70922852 , 23.50069427], crs=ccrs.PlateCarree())#Kolkata
    #ax.set_extent([85.172233, 86.47032928, 19.74252319 , 20.95955657], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m",linewidth=1)
    #ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='k', facecolor='grey'))
    gl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   linewidth=0, color='black', alpha=0.5, linestyle='--')
    gl.top_labels=False
    gl.right_labels=False
    gl.bottom_labels=False
    gl.left_labels=False
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
   



# Show the plots
plt.rcParams['axes.linewidth']=2
plt.rcParams['patch.linewidth']=2
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['xtick.major.size']=12
plt.rcParams['ytick.major.size']=12
plt.rcParams['xtick.minor.size']=4
plt.rcParams['ytick.minor.size']=4
plt.savefig(r"C:\Users\lenovo\Desktop\SIP_IMD_2023\MYJ_Plots\Rainfall\2015042100.jpg", dpi=300)
plt.show()
