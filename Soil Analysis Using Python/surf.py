#!/usr/bin/python3

import pandas as pd
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.basemap import Basemap

data = pd.read_csv('soilData.csv')
data1 = pd.read_csv('elevation.csv')
df = data.set_index('ID')

data = np.array(df)
data1 = np.array(data1)
lat = data[:,0]
long = data[:,1]
elev = data1[:,1]

#Plotting elevation against LAT and LONG

fig0 = plt.figure(figsize=(20,10))
plt.clf()
fig0.canvas.set_window_title('Variation Of Elevation with Latitude and Longitude')
plt.subplots_adjust(hspace = 0.5)
plt.subplot(2, 1, 1)
plt.scatter(lat,elev)
plt.title('Variation of Elevation with Latitude')
plt.xlabel('Latitude')
plt.ylabel('Eleation From Mean Sea Level')

plt.subplot(2, 1, 2)
plt.scatter(long , elev)
plt.title('Variation of Elevation with Longitude')
plt.xlabel('Longitude')
plt.ylabel('Eleation From Mean Sea Level')

#Surface Plot

fig1 = plt.figure(figsize=(20,10))
plt.clf()
fig1.canvas.set_window_title('3D projection of the Surface')
ax = fig1.add_subplot(111, projection='3d')
x = np.linspace(np.min(lat),np.max(lat))
y = np.linspace(np.min(long),np.max(long))
xI,yI = np.meshgrid(x,y)

zI = griddata(lat,long,elev,xI,yI,interp='linear')
ax.plot_surface(xI,yI,zI, rstride=2, cstride=2,alpha=0.8)
ax.set_title('Surface')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Elevation From Mean Sea Level')


#Plot of Locations on a map

from mpl_toolkits.basemap import Basemap
fig2 = plt.figure()
themap = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_0=23.5,lon_0=79.5)

themap.drawcoastlines()
themap.drawcountries()
themap.fillcontinents(color = 'gainsboro')
themap.drawmapboundary(fill_color='steelblue')
parallels = np.arange(0.,81,10.)
themap.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
themap.drawmeridians(meridians,labels=[True,False,False,True])
x, y = themap(long, lat)
themap.plot(x, y, 'o',color='Indigo',markersize=4)

plt.show()

