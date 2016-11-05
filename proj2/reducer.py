#!/usr/bin/python

import sys
# from sklearn.decomposition import PCA

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

def getNumClustersK(centroidFile):
    centroidReader = open(outputFile)
    line = centroidReader.readline()
    k = 1
    while line != "":
        k+=1
        line = centroidReader.readline()
    return k

# dataFile = "dummyData.txt"
# clusterAssignmentFile = "dummyAssignments.txt"
# centroidFile = "dummyCentroids.txt"

file = 1
if (file == 0):
    dataFile = "cho.txt"
    clusterAssignmentFile = "choAssignments.txt"
    centroidFile = "choCentroids.txt"
    outputFile = "choOutput.txt"
elif(file == 1): 
    dataFile = "iyer.txt"
    clusterAssignmentFile = "iyerAssignments.txt"
    centroidFile = "iyerCentroids.txt"
    outputFile = "iyerOutput.txt"
elif (file == 2):
    dataFile = "new_dataset_1.txt"
    clusterAssignmentFile = "new1Assignments.txt"
    centroidFile = "new1Centroids.txt"
    outputFile = "new1Output.txt"
else:
    dataFile = "new_dataset_2.txt"
    clusterAssignmentFile = "new2Assignments.txt"
    centroidFile = "new2Centroids.txt"
    outputFile = "new2Output.txt"


ogData = getOriginalData(dataFile)
oldClusterAssignments = getOldClusterAssignments(clusterAssignmentFile)
k = getNumClustersK(centroidFile)

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

toWriteToOutput = [0]*len(clusterAssignments)
outputWriter = open(outputFile, 'w+')

isSame = compareClusters(oldClusterAssignments, clusterAssignments)
if (isSame):
    #somehow terminate map reduce
    for clusterAssignment in clusterAssignments:
        toWriteToOutput[clusterAssignment[0]-1] = clusterAssignment[1]
        clust = str('\t'.join([str(x) for x in clusterAssignment]))
else:
    # Write new assignments
    assignmentWriter = open(clusterAssignmentFile, 'w+')
    for clusterAssignment in clusterAssignments:
        toWriteToOutput[clusterAssignment[0]-1] = clusterAssignment[1]
        assignmentWriter.write('\t'.join([str(x) for x in clusterAssignment]) + '\n')
    
    writeToCentroidFile = open(centroidFile, 'w+')
    for centroid in newCentroids:
        writeToCentroidFile.write('\t'.join([str(x) for x in centroid]) + '\n')

outputWriter.write(' '.join([str(x) for x in toWriteToOutput]))
    #write to old cluster assignments and add your new cluster assignments 
