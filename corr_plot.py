# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:09:23 2023

@author: lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import os
from scipy.interpolate import interp1d
import seaborn as sns

# Specify the folder path where observed data files are located

# Load and normalize the observed data for each day
resampled_observed_data = []
for day in range(1,13):
    observed_file = os.path.join(r"C:\Users\lenovo\Desktop\SIP_IMD_2023\METAR\humidity\KOL", f'observed_{day}.csv')
    observed_df = pd.read_csv(observed_file)

    # Assuming the observed data is stored in a column named 'Temp'
    observed_values = observed_df['relh'].values

    resampled_observed_data.extend(observed_values)

# Load the simulated data for each day
simulated_data = []
for day in range(1,13):
    simulated_file = os.path.join(r"C:\Users\lenovo\Desktop\no_urbfrac\KOL\RH", f'simulated_{day}.nc')
    simulated_nc = Dataset(simulated_file, 'r')

    for i in range(97):
        simulated_values = simulated_nc.variables['rh'][i,132:182,169:219]
        temp_sum = np.sum(simulated_values)
        ave_sum = temp_sum / 2500
        simulated_data.append(ave_sum)

# Resample the simulated data with 30-minute intervals and length of 48*13
resampled_simulated_data = [simulated_data[i] for i in range(0,1163,2)]
resampled_simulated_data=resampled_simulated_data[:568]

# Combine observed and simulated data into a single DataFrame
data = pd.DataFrame({'Observed': resampled_observed_data,
                     'Simulated': resampled_simulated_data})

# Create a regression plot
sns.lmplot(x='Simulated', y='Observed', data=data, height=8, aspect=1.2, scatter_kws={'s': 30})
plt.yticks([40,60,80,100],fontsize=20)
plt.xticks([25,50,75,100],fontsize=20)
plt.xlabel('RH2 (%)',fontsize=30)
plt.ylabel('RH2 OBS(%)',fontsize=30)
plt.xlabel_style = {'size': 100, 'color': 'black'}
plt.ylabel_style = {'size': 100, 'color': 'black'}
plt.title('Correlation Plot for KOL RH2_CTL',fontsize=30)

# Convert the lists to NumPy arrays for calculations
resampled_observed_data = np.array(resampled_observed_data)
resampled_simulated_data = np.array(resampled_simulated_data)

# Calculate RMSE
rmse = np.sqrt(np.mean((resampled_observed_data - resampled_simulated_data) ** 2))

print('Root mean square error: ',rmse)
#calculate MAE
mae = np.mean(np.abs(resampled_observed_data - resampled_simulated_data))

print('Mean absolute error: ',mae)

# Calculate correlation coefficient
# 

correlation_coefficient = np.ma.corrcoef(resampled_simulated_data, resampled_simulated_data)[0, 1]
print('Correlation coeffecient: ',correlation_coefficient)























# Read the observed and simulated rainfall data into DataFrames
# Reading the observed data in Excel file
# data_obs = pd.read_csv(r"C:\Users\lenovo\Desktop\SIP_IMD_2023\METAR\Kolkata\Temp\2015051600.csv")
# data_obs['datetime'] = pd.to_datetime(data_obs['valid'])
# data_obs = data_obs.set_index('datetime').sort_index()
# #data_obs = data_obs['tmpc']
# data_obs = data_obs['Temp']

# import xarray as xr

# # Read the NetCDF file
# simulated = xr.open_dataset(r"C:\Users\lenovo\Desktop\no_urbfrac\Temp\2015042100_ts_kol_d03_temp2m.nc")

# # Assuming the 15-minute data is in a variable named 'value'
# # Adjust the variable name based on your actual data
# #simulated = simulated['temp']

# #simulated['time'] = pd.to_datetime(simulated.time.values)

# # Align the datasets based on nearest timestamps
# #obs_data_new = data_obs.-reindex(simulated.time, method='nearest')#observed 
# obs_data_resampled = data_obs.resample('15T').ffill()
# #obs_data_normalized = obs_data_resampled / simulated[:len(obs_data_resampled)]
# temps=[]

# for i in range(len(simulated.Time)):
    
#     temp_data = simulated['temp'][i, ...]  # Access the 'temp' variable for each time step

#     sum_of_array = np.sum(temp_data)
#     temps.append(sum_of_array)

# # Divide each sum by 90000
# temps = [temp / 90000 for temp in temps]
# temps=temps[:95]
# #Celtemps=[x - 273 for x in temps]
# temps_new = np.array(temps, dtype=np.float64).tolist()


# # Create a DataFrame with observed and simulated rainfall values
# data = pd.DataFrame({'Observed': obs_data_resampled,'Simulated': temps_new})#temps is float32 and obs_data is float64
# #data=data[49:56]
# # Create the scatter plot with trendline
# plt.figure(figsize=(8, 6))
# sns.regplot(x='Simulated', y='Observed', data=data, scatter_kws={'alpha': 0.8})
# plt.title('Scatter Plot: Observed vs Simulated Temperature'+str())
# # plt.yticks([295,300,305,310,315])
# # plt.xticks([295,300,305,310,315])

# plt.xlabel('Simulated Temp')
# plt.ylabel('Observed Temp')
# plt.show()

# # # Calculate the correlation coefficient
# # correlation_coefficient = obs_data_resampled.corr(simulated_new)

# # # Display the correlation coefficient
# # print('Correlation Coefficient:', correlation_coefficient)

# obs_data_resampled_df =pd.DataFrame(obs_data_resampled)
# simulated_new_df = pd.DataFrame(temps)


# correlation_coefficient = simulated_new_df.corrwith(obs_data_resampled_df)

# print('Correlation Coefficient:', correlation_coefficient)

