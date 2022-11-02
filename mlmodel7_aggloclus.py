import numpy as np
import pandas as pd

import scipy
import pylab

import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix

from matplotlib import pyplot as plt

from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics.pairwise import euclidean_distances
# ----------------------------------------------------------------------------------------------------------------------
# DEBUGGING FOR UNCONDENSED DISTANCE MATRIX
# It means X1 is too close to X1.T
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter

simplefilter("ignore", ClusterWarning)

# OR
from scipy.spatial.distance import squareform
# ----------------------------------------------------------------------------------------------------------------------
# RANDOM DATA
X1, y1 = make_blobs(n_samples=50, centers=[[4, 4], [-2, -1], [1, 1], [10, 4]], cluster_std=0.9)
# plt.scatter(X1[:, 0], X1[:, 1], marker='o')
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# CLUSTERING
agglom = AgglomerativeClustering(n_clusters=4, linkage='average')
agglom.fit(X1, y1)

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6, 4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1
X1 = (X1 - x_min) / (x_max - x_min)

# This loop displays all the datapoints
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
# plt.axis('off')

# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
# plt.show()

dist_matrix = distance_matrix(X1, X1)
# print(dist_matrix)
# print(dist_matrix.shape)
# ----------------------------------------------------------------------------------------------------------------------
# FOR CONDENSING A DISTANCE MATRIX
# condensed_dist_matrix = squareform(dist_matrix)
# print(condensed_dist_matrix.shape)
# ----------------------------------------------------------------------------------------------------------------------
Z = hierarchy.linkage(dist_matrix, 'complete')
dendrogram_1 = hierarchy.dendrogram(Z)
# plt.show()
# Z = hierarchy.linkage(condensed_dist_matrix, 'complete')

Z = hierarchy.linkage(dist_matrix, 'average')
dendrogram_2 = hierarchy.dendrogram(Z)
# plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# EXAMPLE II

filename = 'cars_clus.csv'

# Read csv
pdf = pd.read_csv(filename)
# print("Shape of dataset: ", pdf.shape)

# print(pdf.head(5))

# print("Shape of dataset before cleaning: ", pdf.size)
pdf[['sales', 'resale', 'type', 'price', 'engine_s',
     'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
     'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
                               'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
                               'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
# print("Shape of dataset after cleaning: ", pdf.size)
# print(pdf.head(5))

featureset = pdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

x = featureset.values  # returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
# print(feature_mtx[0:5])

# CALCULATE THE DISTANCE MATRIX
leng = feature_mtx.shape[0]
D = np.zeros([leng, leng])
for i in range(leng):
    for j in range(leng):
        D[i, j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

Z = hierarchy.linkage(D, 'complete')

max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

# k = 5
# clusters = fcluster(Z, k, criterion='maxclust')
# print(clusters)

# DENDROGRAM
fig = pylab.figure(figsize=(18, 50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])))


dendrogram = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')
# plt.show()

# USING SCI-KIT LEARN FOR
dist_matrix = euclidean_distances(feature_mtx,feature_mtx)
print(dist_matrix)

Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')

fig = pylab.figure(figsize=(18, 50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])))


dendro = hierarchy.dendrogram(Z_using_dist_matrix, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12,
                              orientation='right')

agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(dist_matrix)

agglom.labels_

pdf['cluster_'] = agglom.labels_
pdf.head()

import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25)
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

pdf.groupby(['cluster_','type'])['cluster_'].count()

agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
agg_cars

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')