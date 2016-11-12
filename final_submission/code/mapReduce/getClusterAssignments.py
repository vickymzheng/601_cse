#!/usr/bin/python

import sys

def clusterAssignments(args):
	if (len(sys.argv) < 3):
		print "Something went wrong"
		return 
	
	dataFile = sys.argv[1]
	clusterAssignment = sys.argv[2]

	toRead = open(dataFile)
	toWrite = open(clusterAssignment, 'w+')

	line = toRead.readline()
	while line != "":
		entry = line.split()
		entry = entry[:2]
		toWrite.write('\t'.join(entry) + '\n')
		line = toRead.readline()

clusterAssignments(sys.argv)