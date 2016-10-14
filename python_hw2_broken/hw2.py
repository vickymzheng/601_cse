from sets import Set
def convertIntToLabel(i):
	if (i >= 200):
		if i == 200:
			return "ALL"
		elif i == 201:
			return "AML"
		elif i == 202:
			return "Breast Cancer"
		elif i == 203: 
			return "Colon Cancer"
	else:
		stringToReturn = "G"
		stringToReturn = stringToReturn + str(i/2 + 1) + "_"
		if (i%2 == 0):
			stringToReturn +="UP"
		else: 
			stringToReturn +="Down"

		return stringToReturn
def hw2(inputFileName):
	toRead = open(inputFileName)
	line = toRead.readline()
	fileContents = []
	while (line != ""):
		unLabeledData = line.split()
		#Pop off sample label
		unLabeledData.pop(0)
		if len(unLabeledData) == 102:
			#When you split "breast cancer", it makes it two entries
			unLabeledData.pop()
		fileContents.append(unLabeledData)
		line = toRead.readline()

	#This is to clean the data for int comparisons rather than string comparisons 
	for i in range(0, len(fileContents)):
		for j in range(0, len(fileContents[i])):
			if (fileContents[i][j] == "UP"):
				fileContents[i][j] = j*2;
			elif fileContents[i][j] == "Down":
				fileContents[i][j] = j*2+1
			elif fileContents[i][j] == "ALL":
				fileContents[i][j] = 200
			elif fileContents[i][j] == "AML":
				fileContents[i][j] = 201
			elif fileContents[i][j] == "Breast":
				fileContents[i][j] = 202
			elif fileContents[i][j] == "Colon":
				fileContents[i][j] = 203
				
	#Creates an array that holds 204 0's initally
	frequencyCounter = [0]*204 
	for x in fileContents:
		for y in x:
			#This is to count the frequency of each int found. G1_UP will be at 0 so every time we find G1_UP, we increment frequencyCounter[0]
			frequencyCounter[y]+=1

	support = 30
	frequentUnlabeledItemSetsL1 = []
	frequentLabeledItemSetsL1 = {}

	for i in range(0, len(frequencyCounter)):
		if frequencyCounter[i] >= support:
			frequentUnlabeledItemSetsL1.append(i)
			geneLabel = convertIntToLabel(i)
			frequentLabeledItemSetsL1[geneLabel] = frequencyCounter[i]

	frequencyUnlabeledItemSetsL2 = {}
	for i in  range(0, len(frequentUnlabeledItemSetsL1)):
		for j in range(i, len(frequentUnlabeledItemSetsL1)):
			for sample in fileContents:
				firstNum = frequentUnlabeledItemSetsL1[i]
				secondNum = frequentUnlabeledItemSetsL1[j]
				if (firstNum in sample) and (secondNum in sample) and (firstNum != secondNum):
					mapKey = str(firstNum) + " " + str(secondNum) 
					if mapKey in frequencyUnlabeledItemSetsL2:
						frequencyUnlabeledItemSetsL2[mapKey]+=1
					else: 
						frequencyUnlabeledItemSetsL2[mapKey] = 1

	frequentLabeledItemSetL2 = {}
	for x in frequencyUnlabeledItemSetsL2:
		if frequencyUnlabeledItemSetsL2[x] >= support:
			pair = x.split()
			pairAsInt = [int(pair[0]), int(pair[1])]
			firstNum = min(pairAsInt)
			secondNum = max(pairAsInt)
			
			if (firstNum == secondNum):
				print "something went wrong"
			
			mapKey = "[" + convertIntToLabel(firstNum) + ", " + convertIntToLabel(secondNum) + "]"
			frequentLabeledItemSetL2[mapKey] = frequencyUnlabeledItemSetsL2[x]

	
	frequencyUnlabeledItemSetsL3 = {}
	for i in  range(0, len(frequentUnlabeledItemSetsL1)):
		for j in range(i, len(frequentUnlabeledItemSetsL1)):
			for k in range (j, len(frequentUnlabeledItemSetsL1)):
				for sample in fileContents:
					firstNum = frequentUnlabeledItemSetsL1[i]
					secondNum = frequentUnlabeledItemSetsL1[j]
					thirdNum = frequentUnlabeledItemSetsL1[k]
					if (firstNum in sample) and (secondNum in sample) and (thirdNum in sample) and (firstNum != secondNum) and (firstNum != thirdNum) and (secondNum != thirdNum):
						mapKey = str(firstNum) + " " + str(secondNum) + " " + str(thirdNum)
						if mapKey in frequencyUnlabeledItemSetsL3:
							frequencyUnlabeledItemSetsL3[mapKey]+=1
						else: 
							frequencyUnlabeledItemSetsL3[mapKey] = 1

	for x in frequencyUnlabeledItemSetsL3:
		if frequencyUnlabeledItemSetsL3[x] >= support:
			print x
hw2("gene_expression.txt")

