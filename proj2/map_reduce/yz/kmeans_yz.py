import numpy as np
import random
import math
Inputfilename = 'cho.csv'
#Inputfilename = 'iyer.csv'
InputData = np.loadtxt(Inputfilename, delimiter = ',')
InputData = np.insert(InputData, 2, 0, axis = 1) # add a new column to the dataset to store the cluster result

k = 5 #set the number of clusters
center = np.zeros((k,len(InputData[0])))
center_Index = random.sample(xrange(0, len(InputData)),k)
for i in range(0,k):
    center[i] = InputData[center_Index[i]]

center = np.delete(center,[0,1,2],axis = 1)

def calDis(x, center):
    dis = 0                            #calculate the distance between two vectors
    for i in range(3,len(x)):          #skip the first three columuns
        a = x[i] - center[i-3]
        b = math.pow(a,2)
        dis = dis + b
    dis = math.sqrt(dis)
    return dis

def Update(data, center):
    for i in range(0,len(data)):
        disArray = np.zeros(k)
        for j in range(0,len(center)):
            a = calDis(data[i], center[j])
            disArray[j] = a        
        index = np.argmin(disArray)
        #print disArray, index
        data[i][2] = index

def calCenter(data):
    sumArray = np.zeros([k,len(data[0])],dtype = float)
    sumSize = np.zeros(k)
    for i in range(0,len(data)):
        m = int(data[i][2])
        sumArray[m] = sumArray[m] + data[i]
        sumSize[m] = sumSize[m] + 1
    sumArray = np.delete(sumArray,[0,1,2],axis = 1)
    n = int(len(data[0]) - 3)
    center = np.zeros([k,n])
    for j in range(0,k):
        center[j] = sumArray[j] / sumSize[j]
    return center

def checkEql(x,y):
    flag = 0
    if np.array_equal(x,y):
        flag = 1
    return flag
   
flag = 0    
while flag == 0:
    lastC = InputData[:,2]
    Update(InputData,center)
    newC = InputData[:,2]
    center = calCenter(InputData)
    flag = checkEql(lastC,newC)

#print newC
temp =[]
for i in range(0,len(newC)):
    if newC[i] in temp:
        InputData[i][2] = temp.index(newC[i]) + 1
    else:
        temp.append(newC[i])
        InputData[i][2] = len(temp)

print InputData[:,2]
print InputData[:,1]





       
    






                
                
    
       
    

