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

def calcPerformance(samples):
	truePositive = 0.0
	falsePositive = 0.0
	trueNegative = 0.0
	falseNegative = 0.0 
	numSamples = len(samples)
	for sample in samples:
		#sample[-1] is my classification
		#sample[-2] is ground truth
		if (sample[-2] == 1 and sample[-1] == 1):
			truePositive+=1
		elif (sample[-2] == 0 and sample[-1] == 1):
			falsePositive+=1
		elif (sample[-2] == 0 and sample[-1] == 0):
			trueNegative+=1
		else:
			falseNegative+=1

	accuracy = (truePositive+trueNegative)/numSamples
	precision = truePositive / (truePositive + falsePositive)
	recall = truePositive / (truePositive + falseNegative)
	F = (2 * recall * precision) / (recall + precision)

	print "Accuracy: " + str(accuracy)
	print "Precision: " + str(precision)
	print "Recall: " + str(recall)
	print "F: " + str(F)

def randomClassifer(fileName):
	samples = getData(fileName)
	for sample in samples:
		sample[-1] = random.randint(0,1)
		
	calcPerformance(samples)

randomClassifer("project3_dataset2.txt")