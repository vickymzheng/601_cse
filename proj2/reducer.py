#!/usr/bin/python

import sys

def getOldClusterAssignments(clusterAssignmentFile):
    toRead = open(clusterAssignmentFile)
    line = toRead.readline()
    oldClusterAssignments = []
    while line != "":
        clusterAssignment = [int(x) for x in line.split()]
        oldClusterAssignments.append([clusterAssignment[0], clusterAssignment[1]]) 
        line = toRead.readline()
    return oldClusterAssignments

clusterAssignmentFile = "dummyAssignments.txt"
oldClusterAssignments = getOldClusterAssignments(clusterAssignmentFile)

def getNumClusters(clusterAssignments):
    uniqueClusters = []
    for x in clusterAssignments:
        if clusterAssignments[1] not in uniqueClusters:
            uniqueClusters.append(clusterAssignments[1])
    return len(uniqueClusters)

k = getNumClusters(oldClusterAssignments)
clusterAssignments = []
distFromEachCluster = [0]*k

# input comes from STDIN
for line in sys.stdin:
    # parse the input we got from mapper.py
    distances = [float(x) for x in line.split()]
    dataPointID = int(distances[0])
    distances = distances[1:]

    minDist = min(distances)
    indexOfMinDist = distances.index(minDist)
    clusterAssignments.append([dataPointID, indexOfMinDist])

print clusterAssignments

def compareClusters(oldClusterAssignments, clusterAssignments):
    if (len(oldClusterAssignments) != len(clusterAssignments)):
        print "something went wrong"
        return 0

    for clusterAssignment in clusterAssignments:
        if (clusterAssignments[clusterAssignment[0]][1] != oldClusterAssignments[clusterAssignment[0]][1]):
            return 0
    return 1

if (compareClusters(oldClusterAssignments, clusterAssignments) == 1):
    #somehow terminate map reduce
    for clusterAssignment in clusterAssignments:
        print str(clusterAssignment)
else:
    toWrite = open(clusterAssignmentFile, 'w+')
    for clusterAssignment in clusterAssignments:
        toWrite.write('\t'.join([str(x) for x in clusterAssignment]) + '\n')
    #write to old cluster assignments and add your new cluster assignments 
