#!/usr/bin/python3

## This code accesses the Google Elevation API to return the corresponding elevation from sea level using the LAT and LONG array

import geocoder
import pandas as pd
import numpy as np

proxies = {'http':'http://2014239:r12345@172.27.16.154:3128' ,
           'https':'http://2014239:r12345@172.27.16.154:3128',
           'HTTP':'http://2014239:r12345@172.27.16.154:3128',
           'HTTPS':'http://2014239:r12345@172.27.16.154:3128'}

data = pd.read_csv('soilData.csv')
df = data.set_index('ID')

data = np.array(df)
pos = data[:,0:2]

elevation = np.array([])
for i in range(pos.shape[0]):
    g = geocoder.elevation(list(pos[i]),proxies=proxies)
    elevation = np.append(elevation,[g.meters])   

alt = pd.DataFrame({'Elevation' : elevation}) # Creating a DataFrame to store elevation values
alt.to_csv('elevation.csv') #Exporting the data to a CSV
