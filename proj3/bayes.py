# import sys

def addLists(list1, list2):
	#the two lists should be the same length
	numElements = len(list1)
	for index in range(0, numElements):
		list1[index] = list1[index] + list2[index]

def divideList(someList, divisor):
	for index in range(0, len(someList)):
		someList[index] = someList[index]/divisor

def getData(fileName):
	toRead = open(fileName)
	samples = []
	line = toRead.readline()
	while line != "":
	    sample = line.split()
	    myAssignment = -1
	    sample.append(myAssignment)
	    samples.append(sample)
	    line = toRead.readline()
	return samples

def getVariance(numList, mean, numSamples):
	sumSquareDiff = 0
	for sample in numList:
		sumSquareDiff = sumSquareDiff + (mean - sample)**2
	variance = sumSquareDiff/numSamples
	return variance

def isNumeric(potentialNumber):
	try:
		float(potentialNumber)
		return True
	except ValueError:
		return False

def dataPreprocess(data, numSamples, numAttributes):
	for attributeNum in range(0, numAttributes):
		if (isNumeric(data[0][attributeNum])):
			for sampleNum in range(0, numSamples):
				data[sampleNum][attributeNum] = float(data[sampleNum][attributeNum])
		else:
			for sampleNum in range(0, numSamples):
				if data[sampleNum][attributeNum] == "Present":
					data[sampleNum][attributeNum] = 1
				else:
					data[sampleNum][attributeNum] = 0
	return data

def statData(samples, meansPresent, meansAbsent, variancesPresent, variancesAbsent):
	numPresent = 0
	numAbsent = 0
	presentSamples = []
	absentSamples = []
	numAttributes = len(samples[0])-2
	for sample in samples:
		if sample[-2] == 1:
			#this means present
			presentSamples.append(sample)
			numPresent+=1
			#meansPresent =  map(lambda x,y: x + y, meansPresent, sample[0:-2])
			addLists(meansPresent, sample[0:-2])
		else:
			absentSamples.append(sample)
			numAbsent+=1
			#meansAbsent =  map(lambda x,y: x + y, meansAbsent, sample[0:-2])
			addLists(meansAbsent, sample[0:-2])

	#meansPresent = map(lambda x: x/numPresent, meansPresent)
	#meansAbsent = map(lambda x: x/numAbsent, meansAbsent)
	divideList(meansPresent, numPresent)
	divideList(meansAbsent, numAbsent)
	
	for i in range(0, numAttributes):
		presentCol = []
		absentCol = []
		for presentSample in presentSamples:
			presentCol.append(presentSample[i])

		for absentSample in absentSamples:
			absentCol.append(absentSample[i])

		variancesPresent[i] = getVariance(presentCol, meansPresent[i], numPresent)
		variancesAbsent[i] = getVariance(absentCol, meansAbsent[i], numAbsent)

def prior(samples, present, notPresent):
	for sample in samples:
		if sample[-1] == 1:
			present+=1
		if sample[-1] == 0:
			notPresent+=1
	return (present, notPresent)

	return (present, notPresent)
def bayes(fileName):
	samples = getData(fileName)

	numSamples = len(samples)
	numCols = len(samples[0]) 
	dataPreprocess(samples, numSamples, numCols)

	numAttributes = numCols - 2 #removing ground truth column and my assignment column

	meansPresent = [0]*(numAttributes)
	variancesPresent = [0]*(numAttributes)
	meansAbsent = [0]*numAttributes
	variancesAbsent = [0]*numAttributes
	statData(samples, meansPresent, meansAbsent, variancesPresent, variancesAbsent)

	print meansPresent
	print meansAbsent
	print variancesPresent
	print variancesAbsent

	present = 0.0
	notPresent = 0.0
	(present, notPresent) = prior(samples, present, notPresent)
	present = present/numSamples
	notPresent = notPresent/numSamples

	for sample in samples:
		attributes = sample[0:-2]
		attributeProbabilities = []
		for attribute in attributes:
			#calc prob of attribute
			attributeProb = attribute


	

bayes("project3_dataset2.txt")
