def getData(fileName):
	toRead = open(fileName)
	centroids = []
	line = toRead.readline()
	while line != "":
        centroid = [line.split()]
        centroids.append(centroid)
        line = toRead.readline()
    return centroids

data = getData("project3_dataset1.txt")
data = getData("project3_dataset2.txt")