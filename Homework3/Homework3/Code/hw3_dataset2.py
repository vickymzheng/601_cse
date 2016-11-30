# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:12:28 2016

@author: yuzeliu
"""

import numpy as np
#import random
import math
#import sys
from sklearn import preprocessing
import csv

def expand(data,k):
    return np.linalg.matrix_power(data,k)

def inflate(data,r):
    for i in range(0,len(data)):
        for j in range(0,len(data[0])):
            data[i][j] = math.pow(data[i][j],r)
    return preprocessing.normalize(data,norm = 'l1',axis = 0)

def MCL(data,k,r):
    data = expand(data,k)
    data = inflate(data,r)
    return data
        
'''
#for attweb_net.txt, 180 nodes; physics_collaboration_net.txt,142 nodes; yeast, 359 nodes
#Inputfilename = 'attweb_net.txt'
Inputfilename = 'physics_collaboration_net.txt'
#Inputfilename = 'yeast_undirected_metabolic.txt'

dataSet = np.loadtxt(Inputfilename, delimiter = ' ')
if Inputfilename == 'attweb_net.txt':
    assmat = np.zeros([180,180])
elif Inputfilename == 'physics_collaboration_net.txt':
    assmat = np.zeros([142,142])
elif Inputfilename == 'yeast_undirected_metabolic.txt':
    assmat = np.zeros([359,359])
'''

dataSet_pre = np.genfromtxt('physics_collaboration_net.csv', names = None, delimiter=' ', dtype=None)
dataSet = np.zeros([len(dataSet_pre),2])    
assmat = np.zeros([142,142])
narray = []
for i in range(0,len(dataSet_pre)):
    for j in range(0,2):
        if dataSet_pre[i][j] in narray:
            dataSet[i][j] = narray.index(dataSet_pre[i][j])
        else:
            narray.append(dataSet_pre[i][j])
            dataSet[i][j] = len(narray) - 1


for i in range(0,len(dataSet)): #construct associate matrix
    m = int(dataSet[i][0])
    n = int(dataSet[i][1])
    assmat[m][n] = assmat[n][m] = 1

for i in range(0,len(assmat)):  #add selfloop
    assmat[i][i] = 1.0

assmat_norm = preprocessing.normalize(assmat,norm = 'l1',axis = 0)

repeat = 150
k = 6
r = 6
cluster = 1
nodeArray = np.zeros([len(assmat),2])
#flag = 0

for i in range(0,len(assmat)):
    nodeArray[i][0] = i

while repeat > 0:
    assmat = MCL(assmat,k,r)
    repeat = repeat - 1

for i in range(0,len(assmat)):
    for j in range(0,len(assmat)):
        if assmat[i][j] != 0:
            nodeArray[j][1] = cluster
    cluster = cluster + 1


label =[]
temp = []
#nodeArray_name = np.chararray(len(nodeArray),2)

for i in range(0,len(nodeArray)):
    label.append(nodeArray[i][1])
    
for i in range(0,len(label)):
    if label[i] in temp:
        nodeArray[i][1] = temp.index(label[i]) + 1
    else:
        temp.append(label[i])
        nodeArray[i][1] = len(temp)
print 'Index', ' ', 'Name', ' ', 'Cluster'
#for i in range(0,len(nodeArray)):
#    print '{:5}'.format(nodeArray[i][0]), '{:18}'.format(narray[i]), '{:>4}'.format(nodeArray[i][1])
    #print '{} {} {}'.format(nodeArray[i][0],narray[i],nodeArray[i][1])
print '*Vertices 142'
for i in range(0,len(nodeArray)):
    print ' ', int(nodeArray[i][1])
print 'The total number of clusters are : ', len(temp)

    
         

