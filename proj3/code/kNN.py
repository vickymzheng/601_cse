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
    for j in range(0,k):
        for i in range(0,len(v)):
            minimum = sys.maxint
            index = 0
            if v[i] < minimum:
                minimum = v[i]
                index = i
        temp[j] = index
        v[index] = sys.maxint
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
            disMat[i][j] = calDis[testSet[i],trainSet[j]]
    for i in range(0,m):
        nearK = findKmin(disMat[i],k)
        disArray = np.zeros(k)
        classArray = np.zeros(k)
        for j in range(0,k):
            disArray[j]= disMat[i][nearK[j]]
            classArray[j] = trainSet[j][len(trainSet)-2]
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

#Read teh data from the dataset.
Inputfilename = 'project3_dataset1.csv'
#Inputfilename = 'project3_dataset2.csv'
dataSet = np.loadtxt(Inputfilename, delimiter = ',')
dataSet= np.insert(dataSet, len(dataSet[0]), 0, axis = 1) #Add a new columun to store result 
dsetSize = len(dataSet)
np.random.shuffle(dataSet)
ssetSize = int(round(dsetSize/10.0))
for

           
                
            
            
            
            
            
        
        
        
            
            
        





`
