#!/usr/bin/python

import sys
import math

# Each centroid is unlabeled 
centroidFile = "dummyCentroids.txt"
iyer = 0
newData = 1
new1 = 1
if(newData):
    if(new1):
        centroidFile = "new1Centroids.txt"
    else:
        centroidFile = "new2Centroids.txt"
else: 
    if (iyer):
        centroidFile = "iyerCentroids.txt"
    else: 
        centroidFile = "choCentroids.txt"
    
def getCentroids(centroidFile):
    toRead = open(centroidFile)
    centroids = []
    line = toRead.readline()
    while line != "":
        centroid = [float(x) for x in line.split()]
        centroids.append(centroid)
        line = toRead.readline()
    return centroids

global centroids 
centroids = getCentroids(centroidFile)

def getDists(dataPoint):
    # Column 0 in data point is its label
    # Column 1 in data point is ground truth
    # Column 2 - n are the columns of interest
    dists = []
    for centroidIndex in range(0, len(centroids)): 
        dis = 0                            
        for i in range(2,len(dataPoint)):          
            a = dataPoint[i] - centroids[centroidIndex][i-2]
            b = math.pow(a,2)
            dis = dis + b
        dis = math.sqrt(dis)
        dists.append(dis)
    return dists

# input comes from STDIN (standard input)
for line in sys.stdin:

    # Turn string into list of datapoints as floats 
    dataPoint = [float(x) for x in line.split()]
    dataPointID = str(int(dataPoint[0]))

    print dataPointID + '\t' + '\t'.join([str(x) for x in getDists(dataPoint)])
