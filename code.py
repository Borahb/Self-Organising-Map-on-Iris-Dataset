# Self Organising Maps 
"""
Created on Fri Feb 28 08:20:44 2020

@author: Bhaskar
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing the dataset
Dataset = pd.read_csv('iris.csv')
X = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, -1].values

#encoding the y dataset
from sklearn.preprocessing import LabelEncoder
labelencode = LabelEncoder()
y = labelencode.fit_transform(y)


#feature scaling the data
from sklearn.preprocessing import MinMaxScaler
sc =  MinMaxScaler(feature_range = (0,1))
X= sc.fit_transform(X)

#importing minisom
from minisom import MiniSom
som = MiniSom(x = 7, y = 7, input_len = 4, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's', 'D'] 
colors = ['r', 'g','b']# (red,iris-setosa),(green,iris-versicolor),(blue,iris-virginica)
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
plt.axis([0,7,0,7])
show()


