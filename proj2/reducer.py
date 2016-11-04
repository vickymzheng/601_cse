#!/usr/bin/python

import sys

def getOriginalData(dataFile):
    toRead = open(dataFile)
    data = []
    line = toRead.readline()
    while line != "":
        dataEntry = [float(x) for x in line.split()]
        data.append(dataEntry) 
        line = toRead.readline()
    return data

def getOldClusterAssignments(clusterAssignmentFile):
    toRead = open(clusterAssignmentFile)
    line = toRead.readline()
    oldClusterAssignments = []
    while line != "":
        clusterAssignment = [int(x) for x in line.split()]
        oldClusterAssignments.append([clusterAssignment[0], clusterAssignment[1]]) 
        line = toRead.readline()
    return oldClusterAssignments

def getNumClusters(clusterAssignments):
    uniqueClusters = []
    for x in clusterAssignments:
        if x[1] not in uniqueClusters:
            uniqueClusters.append(x[1])
    return len(uniqueClusters)

def compareClusters(oldClusterAssignments, clusterAssignments):
    if (len(oldClusterAssignments) != len(clusterAssignments)):
        print "something went wrong"
        return 0
        
    for clusterAssignment in clusterAssignments:
        if (clusterAssignment[1] != oldClusterAssignments[clusterAssignment[0]-1][1]):
            return 0
    return 1

def newCentroids(clusterMembers,k):
    # Should be k * n, k new centroids with n dimensions 
    numDimensions = len(clusterMembers[1][0])
    newCentroids = [[0 for x in range(numDimensions)] for y in range(k)]
    for cluster in clusterMembers:
        numMembers = len(clusterMembers[cluster])
        for member in clusterMembers[cluster]:
            for index in range(0, len(member)):
                newCentroids[cluster-1][index] = newCentroids[cluster-1][index] + (member[index]/numMembers)
    return newCentroids


# dataFile = "dummyData.txt"
# clusterAssignmentFile = "dummyAssignments.txt"
# centroidFile = "dummyCentroids.txt"
iyer = 0
if (iyer):
    dataFile = "iyer.txt"
    clusterAssignmentFile = "iyerAssignments.txt"
    centroidFile = "iyerCentroids.txt"
    outputFile = "iyerOutput.txt"
else: 
    dataFile = "cho.txt"
    clusterAssignmentFile = "choAssignments.txt"
    centroidFile = "choCentroids.txt"
    outputFile = "choOutput.txt"

ogData = getOriginalData(dataFile)
oldClusterAssignments = getOldClusterAssignments(clusterAssignmentFile)
k = getNumClusters(oldClusterAssignments)

clusterAssignments = []
clusterMembers = {}

# input comes from STDIN
for line in sys.stdin:
    # parse the input we got from mapper.py
    distances = [float(x) for x in line.split()]           
    dataPointID = int(distances[0])
    distances = distances[1:]

    minDist = min(distances)
    indexOfMinDist = distances.index(minDist) + 1 # Having index start at 1 rather than 0 
    clusterAssignments.append([dataPointID, indexOfMinDist])
    if (indexOfMinDist not in clusterMembers):
        clusterMembers[indexOfMinDist] = []
    
    clusterMembers[indexOfMinDist].append(ogData[dataPointID-1][2:])


# Need to calc new centroids here
newCentroids = newCentroids(clusterMembers, k)

isSame = compareClusters(oldClusterAssignments, clusterAssignments)
if (isSame):
    toWrite = open(outputFile, 'w+')
    #somehow terminate map reduce
    toWriteToOutput = [0]*len(clusterAssignments)
    for clusterAssignment in clusterAssignments:
        toWriteToOutput[clusterAssignment[0]-1] = clusterAssignment[1]
        clust = str('\t'.join([str(x) for x in clusterAssignment]))
        print clust
    toWrite.write(' '.join([str(x) for x in toWriteToOutput]))
else:
    # Write new assignments
    toWrite = open(clusterAssignmentFile, 'w+')
    for clusterAssignment in clusterAssignments:
        toWrite.write('\t'.join([str(x) for x in clusterAssignment]) + '\n')
    
    writeToCentroidFile = open(centroidFile, 'w+')
    for centroid in newCentroids:
        writeToCentroidFile.write('\t'.join([str(x) for x in centroid]) + '\n')

    #write to old cluster assignments and add your new cluster assignments 
