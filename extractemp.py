# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:55:58 2023

@author: lenovo
"""


import os
import subprocess
from netCDF4 import Dataset
import wrf
from wrf import getvar
import concurrent.futures

folders=subprocess.check_output('ls', shell=True).decode('utf-8').split("\n")
folders.remove('')

try:
    os.mkdir('temp')
except:
    pass

def extract(fold):
    if '.py' not in fold and 'temp' not in fold:
        print(fold)
        files=subprocess.check_output([f'ls {fold}/wrfout_d03_*'], shell=True).decode('utf-8').split("\n")
        files.remove('')
        file=files[0]

        ncfile = Dataset(file)
        temp=getvar(ncfile,'T2',timeidx=wrf.ALL_TIMES)


        temp.name='temp'
        temp.to_netcdf(f'temp/{fold}_ts_kol_d03_temp2m.nc')
        return None

with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:
    results = executor.map(extract, folders)
