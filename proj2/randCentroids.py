#!/usr/bin/python

import sys
import numpy as np
import random
import math


def getCentroids(args):
	if (len(sys.argv) < 4):
		print "Something went wrong"
		return 
	
	dataFile = sys.argv[1]
	clusterFile = sys.argv[2]
	k = int(sys.argv[3])

	toRead = open(dataFile)
	toWrite = open(clusterFile, 'w+')

	data = []
	line = toRead.readline()
	while line != "":
		entry = line.split()
		entry = entry[2:]
		data.append(entry)
		line = toRead.readline()

	centerIndices = random.sample(xrange(0, len(data)),k)

	for index in centerIndices:
		toWrite.write('\t'.join(data[index]) + '\n')

getCentroids(sys.argv)