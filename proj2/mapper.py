#!/usr/bin/python

import sys
import math

# Each centroid is unlabeled 
centroidFile = "dummyCentroids.txt"
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


fileToOpen = "dummyData.txt"
def getData(fileToOpen):
    toRead = open(fileToOpen)
    data = []
    line = toRead.readline()
    while line != "":
        dataEntry = [float(x) for x in line.split()]
        data.append(dataEntry)
        line = toRead.readline()
    return data

data = getData(fileToOpen)

print map(getDists, data)

# input comes from STDIN (standard input)
# for line in sys.stdin:
#     # remove leading and trailing whitespace
#     line = line.strip()
#     # split the line into words
#     dataPoint = [int(x) for x in line.split()]

#     # increase counters
#     for word in words:
#         # write the results to STDOUT (standard output);
#         # what we output here will be the input for the
#         # Reduce step, i.e. the input for reducer.py
#         #
#         # tab-delimited; the trivial word count is 1
#         print '%s\t%s' % (word, 1)
