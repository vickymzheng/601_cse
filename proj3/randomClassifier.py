import random

def getData(fileName):
	toRead = open(fileName)
	samples = []
	line = toRead.readline()
	while line != "":
	    sample = line.split()
	    sample[-1] = int(sample[-1])
	    myAssignment = -1
	    sample.append(myAssignment)
	    samples.append(sample)
	    line = toRead.readline()
	return samples

def checkCorrect(samples):
	numCorrect = 0.0
	numSamples = len(samples)
	for sample in samples:
		if (sample[-2] == sample[-1]):
			numCorrect+=1

	return numCorrect/numSamples

def randomClassifer(fileName):
	samples = getData(fileName)
	for sample in samples:
		sample[-1] = random.randint(0,1)
		
	print checkCorrect(samples)

randomClassifer("project3_dataset1.txt")