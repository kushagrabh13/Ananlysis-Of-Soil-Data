#!/usr/bin/python3

##The following code creates a model of the soilData using the Distance Weighted K-Nearest Neighbours Algorithm and predicts the subsequent soil property 
##using the LATITUDE and LONGITUDE as features.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
data = pd.read_csv('soilData.csv') # Reading csv data into Panda's Dataframe
df = data.set_index('ID') # Seting the index to 'ID' column of the csv file

data = np.array(df) # copying the data from the dataframe to a NumPy array
inputs = data[:,0:2] # Slicing the first two columns for input (LAT and LONG)
outputs = data[:, 2]

#Split into train and test set

from sklearn.cross_validation import train_test_split
X,X_test,y,y_test = train_test_split(inputs, outputs)

#Weighted KNN
    
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor(weights = 'distance') ##Defining the classifier

##from sklearn.preprocessing import PolynomialFeatures
##poly = PolynomialFeatures(2)
##X = poly.fit_transform(X)
reg.fit(X,y) # Fitting the regressor with Training data
##X_test = poly.fit_transform(X_test)
y_pred = reg.predict(X_test) # Predictions based on test set
print('R^2 Score : ',reg.score(X_test,y_test)) # Printing the R^2 Score
accuracy = (np.sum(1-abs((y_test-y_pred)/y_test))/(y_test.size))*100 # Calculating Accuracy
print('Acuracy : ',accuracy)

#Plot of pH variation along LAT and LONG

fig0 = plt.figure(figsize=(20,10)) #creating a blank figure window
fig0.canvas.set_window_title('Variation Of pH with Latitude and Longitude')
plt.subplots_adjust(hspace = 0.5)
plt.subplot(2, 1, 1) #first subplot
plt.scatter(X[:,0] , y) #Plotting data against LAT in first subplot
plt.title('Variation of pH with Latitude')
plt.xlabel('Latitude')
plt.ylabel('pH')

plt.subplot(2, 1, 2) #Second Subplot
plt.scatter(X[:,1] , y) #Plotting data against LONG in second subplot
plt.title('Variation of pH with Longitude')
plt.xlabel('Longitude')
plt.ylabel('pH')

from matplotlib import cm
from matplotlib.mlab import griddata
import scipy.interpolate as interp

#Plot of hypothesis

fig1 = plt.figure(figsize=(20,10)) #creating second blank figure
fig1.canvas.set_window_title('Hyperplane for predicting pH')
plt.clf() #clear figure
ax = fig1.add_subplot(111, projection='3d') #Defining 3D axes
ax.scatter(X[:,0],X[:,1],y,color='red') #Scatter plot of data against LAT and LONG
x_cor0 = np.linspace(np.min(X_test[:,0]),np.max(X_test[:,0])) #defining a linearly spaced vector between min and max LAT
y_cor0 = np.linspace(np.min(X_test[:,1]),np.max(X_test[:,1])) #defining a linearly spaced vector between min and max LONG
xI0,yI0 = np.meshgrid(x_cor0,y_cor0) #Creating a MeshGrid with LAT and LONG data

zI0 = griddata(X_test[:,0],X_test[:,1],y_pred,xI0,yI0,interp='linear') #Defining Z-axis coordinates to visualize the best fit HyperPlane
ax.plot_surface(xI0,yI0,zI0,rstride=2, cstride=2,alpha=0.5) # Plot the surface of HyperPlane
ax.set_title('Hyperplane')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('pH')

#Plot of Surface
fig2 = plt.figure(figsize=(20,10))
fig2.canvas.set_window_title('Plot of Surface with Altitude')
plt.clf()
ax1 = fig2.add_subplot(111, projection='3d')
ax1.scatter(X[:,0],X[:,1],y,color='red')
data1 = pd.read_csv('elevation.csv') #Reading the elevation csv
data1 = np.array(data1) #copying the dataframe into NumPy array
lat = data[:,0] 
long = data[:,1]
elev = data1[:,1]
elev_scaled = np.array([]) #Defining an empty NdArray for scaling the elevation to pH levels ie from 400-480 to 0-10
for i in range(elev.size): 
    elev_scaled = np.append(elev_scaled, (((elev[i]-elev.min())*(10))/(elev.max()-elev.min())) + (0))
x_cor1 = np.linspace(np.min(lat),np.max(lat))
y_cor1 = np.linspace(np.min(long),np.max(long))
xI1,yI1 = np.meshgrid(x_cor1,y_cor1)

zI1 = griddata(lat,long,elev_scaled,xI1,yI1,interp='linear')

ax1.plot_surface(xI1,yI1,zI1, rstride=2, cstride=2)
ax1.set_title('Surface')
ax1.set_xlabel('Latitude')
ax1.set_ylabel('Longitude')
ax1.set_zlabel('Elevation From Mean Sea Level (Scaled Down to 0-10) ')

plt.show() #Display the Plot

