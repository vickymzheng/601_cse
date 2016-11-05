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

def otherGetData(dataFile):
    gene_num = 0
    exp_num = 0
    if dataFile == "cho.txt":
        gene_num = 386
        exp_num = 16
    elif dataFile == "iyer.txt":
        gene_num = 517
        exp_num = 12
    elif (dataFile == "new_dataset_1.txt"):
        gene_num = 150
        exp_num = 3 
    elif (dataFile == "new_dataset_2.txt"):
        gene_num = 6
        exp_num = 5

    exp_matrix = numpy.zeros((gene_num,exp_num))
    line_num = 0
    with open(dataFile) as f:
        for line in f:
            parts = line.split()
            for i in range ( 2, exp_num+2):
                exp_matrix[line_num][i-2] = float(parts[i])
            line_num += 1
        return exp_matrix

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
    data = numpy.matrix(data).T
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
# 0 = cho
# 1 = iyer
# 2 = new1
# 3 = new2

labelFile = ""
dataFile = ""
file = 1
if (file == 0):
    labelFile = "choOutput.txt"
    dataFile = "cho.txt"
elif (file == 1):
    labelFile = "iyerOutput.txt"
    dataFile = "iyer.txt"
elif (file == 2):
    labelFile = "new1Output.txt"
    dataFile = "new_dataset_1.txt"
else:
    labelFile = "new2Output.txt"
    dataFile = "new_dataset_2.txt"

labelFileToRead = open(labelFile)
line = labelFileToRead.readline()
label = [int(x) for x in line.split()]

data = getData(dataFile)
pca_visual(data, label, 2, 4)