#!/usr/bin/python

import sys
def copyFile(args):
	if (len(sys.argv) < 3):
		print "Something went wrong"
		return 
	
	fileToCopy = sys.argv[1]
	fileToCopyTo = sys.argv[2]

	toRead = open(fileToCopy)
	toWrite = open(fileToCopyTo, 'w+')

	line = toRead.readline()
	while line != "":
		toWrite.write(line + '\n')
		line = toRead.readline()

copyFile(sys.argv)