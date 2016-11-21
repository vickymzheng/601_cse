def partitionData(data):
	numSamples = len(data)
	numAttributes = len(data[0])


	for attributeNum in range(0, numAttributes):
		if (isNumeric(data[0][attributeNum])):
			for sampleNum in range(0, numSamples):
				data[sampleNum][attributeNum] = float(data[sampleNum][attributeNum])		

	return data

def getData(fileName):
	toRead = open(fileName)
	centroids = []
	line = toRead.readline()
	while line != "":
	    centroid = line.split()
	    centroids.append(centroid)
	    line = toRead.readline()

	centroids = partitionData(centroids)
	
	return centroids

def isNumeric(potentialNumber):
	try:
		float(potentialNumber)
		return True
	except ValueError:
		return False

def bayes(fileName):
	data = getData(fileName)


# bayes("project3_dataset1.txt")
#bayes("project3_dataset2.txt")
getData("project3_dataset2.txt")
