# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 09:44:55 2018

@author: z3332903
"""

# one layer neural net to calibrate magnetometer

"""mCal = gain(mMeasured+bias)"""

import numpy as np
import numpy.random
import random
import matplotlib
from matplotlib import pyplot as plt
import os
import pandas as pd
import math

cwd = os.getcwd()

print (os.listdir('.'))

import NN_MagCal_DataSetup

learningRate = 0.1
batchSize = 25
trainingSize = 2000
testSize = 200
epochs = 20
batchesPerEpoch = 100

data = []

trainingSet = [] # each data point [[magMeasured],[bMeasured]]
testSet = []

losses = [] # losses to plot

gains = np.random.normal(0.95,0.15, (3,1)) # initialise gains vector - the weights in the neural net

#these biases need to be divided by the gains        
biases = np.random.normal(0,15, (3,1)) # initialise biases vector - the biases in the neural net - 

data = NN_MagCal_DataSetup.data

"""

for e in range(epochs):
    
    for b in range(batchesPerEpoch):
        
        #store batch gradients
        batchDeltaGains = []
        batchDeltaBiases = []

        for i in range(batchSize):
            
            a = random.randint(0,len(trainingSet))
        
            magMeasured = trainingSet[a][0] #np.array([[1],[1],[1]]) # measured mag values                   
        
            magCal = (magMeasured*gains) + biases # magCal calculation
        
            bPred = (np.sum(np.square(magCal)))**0.5 #root sum square of magCal
        
            bMeasured = trainingSet[a][1] # measure value of b
        
            lossDelta = bPred - bMeasured # calculate difference between predicated and 'measured' - also the derivative of dl/db
        
            loss = 0.5*(bPred - bMeasured)**2 # loss function
            
            losses.append(loss)
        
            dbdm = magCal/bPred # dbdm
        
            deltaBiases = lossDelta*dbdm # gradient of the biases
        
            deltaGains = deltaBiases*magMeasured # gradient of the gains
        
            batchDeltaGains.append(deltaGains) # add gradients to batch set
            batchDeltaBiases.append(deltaBiases)
        
        #for each batch probably calculate an average gradient
        
        batchDeltaGainsAvg = np.average(batchDeltaGains)
        batchDeltaBiasesAvg = np.average(batchDeltaBiases) 
        
        #backprop the average gradients
        gains -= deltaGains*learningRate
        biases -= deltaBiases*learningRate

plt.plot(losses)
plt.show()

"""
    





