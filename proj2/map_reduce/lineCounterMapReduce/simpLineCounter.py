toRead = open("cho.txt")
line = toRead.readline()
count = 0
while (line != ""):
	count+=1
	line = toRead.readline()
print count