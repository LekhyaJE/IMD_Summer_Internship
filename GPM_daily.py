# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:54:18 2023

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

folder_path = r"C:\Users\lenovo\Desktop\SIP_IMD_2023\GPM_data\Kolkata\2016050700"
file_pattern = '*.nc4'


for file in glob.glob('*nc'):
    print(file)