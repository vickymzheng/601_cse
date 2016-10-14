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
			frequencyCounter[y]+=1

	support = 30
	frequentUnlabeledItemSetsL1 = []
	frequentLabeledItemSetsL1 = {}

	for i in range(0, len(frequencyCounter)):
		if frequencyCounter[i] >= support:
			frequentUnlabeledItemSetsL1.append(i)
			geneLabel = convertIntToLabel(i)
			frequentLabeledItemSetsL1[geneLabel] = frequencyCounter[i]

hw2("gene_expression.txt")