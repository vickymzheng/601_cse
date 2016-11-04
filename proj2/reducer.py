#!/usr/bin/python

import sys

k = 3
clusterAssignments = {}
distFromEachCluster = [0]*k

# input comes from STDIN
for line in sys.stdin:
    # parse the input we got from mapper.py
    distances = [float(x) for x in line.split()]
    dataPointID = int(distances[0])
    distances = distances[1:]

    minDist = min(distances)
    indexOfMinDist = distances.index(minDist)
    print indexOfMinDist
