TASK-2: PREDICTION USING UNSUPERVISED ML

# Importing the dataset and the required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data = pd.read_csv('/content/Iris.csv')

# Understanding the dataset

data.head()

data.tail()

data.shape

data.info()

# Checking for NULL values

data.isna().sum()

# Summary of the dataset

data.describe()

# Finding all unique values in SepalLengthCm

data["SepalLengthCm"].unique()

# Finding all unique values in SepalWidthCm

data["SepalWidthCm"].unique()

# Finding all unique values in PetalLengthCm

data["PetalLengthCm"].unique()

# Finding all unique values in PetalWidthCm

data["PetalWidthCm"].unique()

# Counting the number of species in dataset

data.groupby(["Species"]).count()

sns.countplot(x='Species',data=data)
plt.title('Species',fontsize=20)
plt.show()

# From pairplot, we find the different combinations of plots between the species

sns.pairplot(hue='Species',data=data)
plt.show()

# Implementation of K-means clustering

X = data.iloc[:, [1,2,3,4]].values
print(X)

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title(' Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

From this, we obtain the no. of clusters= 3

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

Finding the clusters and its centroids by K-means

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'magenta', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 200, c = 'darkred', label = 'Centroids')
plt.title('Clusters')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()

# Therefore, from this we conclude that the optimal number of clusters are 3
