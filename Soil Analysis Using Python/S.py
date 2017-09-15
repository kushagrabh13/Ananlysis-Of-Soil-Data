#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
data = pd.read_csv('soilData.csv')
df = data.set_index('ID')

#Outliers

##df = df[df['S(mgkg-1)'] < 30]

data = np.array(df)
inputs = data[:,0:2]
outputs = data[:,6]

#Split into train and test set#

from sklearn.cross_validation import train_test_split
X,X_test,y,y_test = train_test_split(inputs, outputs)

#Weighted KNN
    
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor(weights = 'distance')
reg.fit(X,y)
y_pred = reg.predict(X_test)
print('R^2 Score : ',reg.score(X_test,y_test))
accuracy = (np.sum(1-abs((y_test-y_pred)/y_test))/(y_test.size))*100
print('Acuracy : ',accuracy)

#Plot of S variation along LAT and LONG

fig0 = plt.figure(figsize=(20,10))
fig0.canvas.set_window_title('Variation Of S with Latitude and Longitude')
plt.subplots_adjust(hspace = 0.5)
plt.subplot(2, 1, 1)
plt.scatter(X[:,0] , y)
plt.title('Variation of S with Latitude')
plt.xlabel('Latitude')
plt.ylabel('S')

plt.subplot(2, 1, 2)
plt.scatter(X[:,1] , y)
plt.title('Variation of S with Longitude')
plt.xlabel('Longitude')
plt.ylabel('S(mgkg-1)')

from matplotlib import cm
from matplotlib.mlab import griddata
import scipy.interpolate as interp

#Plot of hypothesis

fig1 = plt.figure(figsize=(20,10))
fig1.canvas.set_window_title('Hyperplane for predicting S')
plt.clf()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],y,color='red')
x_cor0 = np.linspace(np.min(X_test[:,0]),np.max(X_test[:,0]))
y_cor0 = np.linspace(np.min(X_test[:,1]),np.max(X_test[:,1]))
xI0,yI0 = np.meshgrid(x_cor0,y_cor0)

zI0 = griddata(X_test[:,0],X_test[:,1],y_pred,xI0,yI0,interp='linear')
ax.plot_surface(xI0,yI0,zI0,rstride=2, cstride=2,alpha=0.5)
ax.set_title('Hyperplane')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('S(mgkg-1)')

#Plot of Surface
fig2 = plt.figure(figsize=(20,10))
fig2.canvas.set_window_title('Plot of Surface with Altitude')
plt.clf()
ax1 = fig2.add_subplot(111, projection='3d')
ax1.scatter(X[:,0],X[:,1],y,color='red')
data1 = pd.read_csv('elevation.csv')
data1 = np.array(data1)
lat = data[:,0]
long = data[:,1]
elev = data1[:,1]
elev_scaled = np.array([])
for i in range(elev.size):
    elev_scaled = np.append(elev_scaled, (((elev[i]-elev.min())*(70))/(elev.max()-elev.min())) + (0))
x_cor1 = np.linspace(np.min(lat),np.max(lat))
y_cor1 = np.linspace(np.min(long),np.max(long))
xI1,yI1 = np.meshgrid(x_cor1,y_cor1)

zI1 = griddata(lat,long,elev_scaled,xI1,yI1,interp='linear')

ax1.plot_surface(xI1,yI1,zI1, rstride=2, cstride=2,alpha=0.5)
ax1.set_title('Surface')
ax1.set_xlabel('Latitude')
ax1.set_ylabel('Longitude')
ax1.set_zlabel('Elevation From Mean Sea Level (Scaled Down to 0-70) ')

plt.show()

