# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:47:16 2016

@author: yuzeliu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:38:22 2016

@author: yuzeliu
"""
#CSE601 Datamining and Bio-informatics Project3 - Nearest Neighbor Algorithm Version1
import numpy as np
#import random
import math
import sys
import csv


#-----------------------------------------------------------#
#calculate the Euclidian distance
def calDis(v1,v2):
    sumdis = 0
    for i in range(0,len(v1)-2):
        a = v1[i] - v2[i]
        b = math.pow(a,2)
        sumdis = sumdis + b
    sumdis = math.sqrt(sumdis)
    return sumdis
#-----------------------------------------------------------#
    
#-----------------------------------------------------------#  
#Function for finning the k miimum value in an array        
def findKmin(v,k):               
    temp = np.zeros(k)
    temp2 = np.zeros(k)
    for j in range(0,k):
        minimum = sys.maxint
        index = 0
        for i in range(0,len(v)):
            if v[i] < minimum:
                minimum = v[i]
                index = i
        temp[j] = int(index)
        temp2[j] = minimum
        v[index] = sys.maxint
    for j in range(0,k):
        v[int(temp[j])] = temp2[j]
    return temp
#-----------------------------------------------------------#

#-----------------------------------------------------------#        
# Function of K nearest neighbours algorithm
def KNN(trainSet, testSet):
    k = 3                            #find the kth minimum number in the array
    m = len(testSet)
    n = len(trainSet)
    disMat = np.zeros([m,n])
    for i in range(0,len(testSet)):
        for j in range(0,len(trainSet)):
            disMat[i][j] = calDis(testSet[i],trainSet[j])      
    for i in range(0,m):
        nearK = findKmin(disMat[i],k)
        disArray = np.zeros(k)
        classArray = np.zeros(k)
        for j in range(0,k):
            disArray[j]= disMat[i][int(nearK[j])]
            classArray[j] = trainSet[int(nearK[j])][len(trainSet[0])-2]
        weight0 = 0
        weight1 = 0
        for p in range(0,k):
            if classArray[p] == 0:
                temp = math.pow(disArray[p],2)
                weight0 = weight0 + 1 / temp
            elif classArray[p] == 1:
                temp = math.pow(disArray[p],2)
                weight1 = weight1 + 1 / temp      
        if weight0 > weight1:
            testSet[i][len(testSet[0])-1] = 0
        elif weight0 < weight1:
            testSet[i][len(testSet[0])-1] = 1
#-----------------------------------------------------------#            

#-----------------------------------------------------------# 
#Implement 10-fold cross validation
'''
#Read the data from the dataset.
Inputfilename = 'project3_dataset1.csv'
dataSet = np.loadtxt(Inputfilename, delimiter = ',')
dataSet= np.insert(dataSet, len(dataSet[0]), 5, axis = 1) #Add a new columun to store result 
dsetSize = len(dataSet)
np.random.shuffle(dataSet)
ssetSize = int(round(dsetSize/10.0))
'''
filename = 'project3_dataset2.csv'
with open(filename) as f:
    reader = csv.reader(f)
    dataSet = list(reader)
    
for i in range(0, len(dataSet)):
    for j in range(0,len(dataSet[0])):
        if dataSet[i][j] == 'Present':
            dataSet[i][j] = 1
        elif dataSet[i][j] == 'Absent':
            dataSet[i][j] = 0
            
with open('processed.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(dataSet)

Inputfilename = 'processed.csv'
dataSet = np.loadtxt(Inputfilename, delimiter = ',')
dataSet= np.insert(dataSet, len(dataSet[0]), 5, axis = 1) #Add a new columun to store result 
dsetSize = len(dataSet)
np.random.shuffle(dataSet)
ssetSize = int(round(dsetSize/10.0))    
        
for i in range(0,9):
    start = i * ssetSize 
    end = start + ssetSize
    testSet = dataSet[start:end]
    trainSet = np.delete(dataSet,np.s_[start:end],0)
    KNN(trainSet,testSet)

testSet = dataSet[end:]
trainSet = np.delete(dataSet,np.s_[end:],0)
KNN(trainSet,testSet)

#Performance Evaluation
a = float(0)
b = float(0)
c = float(0)
d = float(0)
for i in range(0,len(dataSet)):
    if dataSet[i][len(dataSet[0])-2] == 1 and dataSet[i][len(dataSet[0])-1] == 1:
        a = a + 1
    elif dataSet[i][len(dataSet[0])-2] == 0 and dataSet[i][len(dataSet[0])-1] == 0:
        d = d + 1
    elif dataSet[i][len(dataSet[0])-2] == 1 and dataSet[i][len(dataSet[0])-1] == 0:
        b = b + 1
    elif dataSet[i][len(dataSet[0])-2] == 0 and dataSet[i][len(dataSet[0])-1] == 1:
        c = c + 1

      
Accuracy = (a+d)/(a+b+c+d)
p = a / (a+c)
r = a / (a+b)
F = (2 * r * p) / (r + p)

print 'Accuracy : ', Accuracy
print 'Precision : ',  p
print 'Recall : ', r
print 'F-measure : ' ,F              

    

           
                
            
            
            
            
            
        
        
        
            
            
        





