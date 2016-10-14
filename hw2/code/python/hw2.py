def hw2(inputFileName):
	toRead = open(inputFileName)
	line = toRead.readline()
	fileContents = []
	while (line != ""):
		unLabeledData = line.split()
		unLabeledData.pop(0)
		if len(unLabeledData) == 102:
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
				
	


hw2("gene_expression.txt")