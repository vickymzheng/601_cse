import numpy
from scipy.spatial.distance import pdist, squareform
import math
from scipy.linalg import sqrtm
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import imp
imp.load_source('proj3d', '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/mpl_toolkits/mplot3d/proj3d.py')
imp.load_source('art3d', '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/mpl_toolkits/mplot3d/art3d.py')
imp.load_source('axis3d', '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/mpl_toolkits/mplot3d/axis3d.py')
imp.load_source('Axes3D', '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/mpl_toolkits/mplot3d/axes3d.py')

def getData(dataFile):
    dataFileToRead = open(dataFile)
    data = []
    line = dataFileToRead.readline()
    while line != "":
        dataEntry = [float(x) for x in line.split()]
        data.append(dataEntry[2:])
        line = dataFileToRead.readline()
    return data

def pca_visual(data,label,dim,algo): # data is sample_num * feature_num
    pca = PCA(n_components=dim)
    data = numpy.matrix(data)
    print data
    data = data.T
    pca.fit(data)
    data_pca = pca.components_
    data_pca = data_pca
    fig = plt.figure()
    if algo  == 1:
        title = 'Kmeans PCA scatter results'
    elif algo == 2:
        title = 'Hierarchical clustering PCA scatter results'
    elif algo == 3:
        title = 'Spectral clustering PCA scatter results'
    elif algo == 4:
        title = 'Map Reduce Kmeans PCA scatter results'

    if dim == 3:
        ax = fig.gca(projection='3d')
        ax.scatter(data_pca[0,],data_pca[1,], data_pca[2,], c=label,marker='o',s=30)
        ax.set_title(title)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

    elif dim == 2:
        ax = fig.gca()
        ax.scatter(data_pca[0,],data_pca[1,], c=label,marker='o',s=30)
        ax.set_title(title)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
    
    plt.grid()
    plt.show()
    return pca

labelFile = "new1Output.txt"
labelFileToRead = open(labelFile)
line = labelFileToRead.readline()
label = [int(x) for x in line.split()]

dataFile = "new_dataset_1.txt"
data = getData(dataFile)
pca_visual(data, label, 2, 4)