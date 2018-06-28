#!/usr/bin/env python
import numpy as np

#ReadFile
trainData =np.genfromtxt('1-prostate-training-data.csv', delimiter=',').astype(float)


# Get row column from Training data
rowT = len(list(trainData))
colT = trainData.shape[1]

I = np.eye(colT) #Unit matrix

def W(lamda,A,Y):
   return np.dot(np.linalg.pinv(np.dot(A.T ,A) +np.dot(lamda,I)), np.dot(A.T, Y))

#LossFunction
def RSS(lamda, A,Y):
    return 0.5*np.linalg.norm(Y-np.dot(A,W(lamda,A,Y))) + lamda*np.linalg.norm(W(lamda,A,Y))

# Get Validation Data from 1/4 Training Data
rowV = rowT/4

# Init Matrix A,Y Validation Data
Av = trainData[0:rowV, 0:colT-1]
Av= np.concatenate((np.ones((rowV, 1)), Av),axis=1)
Yv = trainData[0:rowV, colT-1:colT]

#Find lamda function:
def findLamda(lamdaStart,lamdaEnd,step):
    lamdaList = []
    rssList = [] 
    print("Xrange: "), lamdaStart, lamdaEnd, step
 
    #For(lamdai = lamdaStart, lamdai<lamdaEnd, lamdai+= step) Run lamda from lamdaStart to lamdaEnd
    lamdai = lamdaStart
    while(lamdai <lamdaEnd):
        rss = RSS(lamdai,Av,Yv)
        lamdaList.append(lamdai)
        rssList.append(rss)
        print "rss= ", rss, "  lamda= ", lamdai
        lamdai+= step
    # evalute minRss --> found lamda
    minRssIndex = np.argmin(rssList)
    minRss = rssList[minRssIndex]
    lamdaFound = lamdaList[minRssIndex]
    print "minrss= ", minRss, "   lamda= ", lamdaFound
    print "  "
    
    # stop when rss small enough
    if(minRss < rssCeil):
        return lamdaFound
    else:
        # if(lamdaFound-step <lamdaFloor):
        #     lamdaStart=lamdaFloor
        # else:
        #     lamdaStart= lamdaFound -step
        lamdaStart=lamdaFound
        findLamda(lamdaStart, lamdaFound+step, step/10)



# lamdaFloor = 0.001
rssCeil = 1
lamda = findLamda(0.001, 100, 9)
print "Founded Lamda: ", lamda 

# INIT MATRIX A train, Y train
At = trainData[rowV:rowT, 0:colT-1] # Get row remaining
At = np.concatenate((np.ones((At.shape[0], 1)), At), axis = 1) #add col 1 in index 0 
Yt = trainData[rowV:rowT, colT-1:colT] # get end col.

# Evalute w
w= W(lamda, At, Yt)

print "w = ", w

#Get data from file need prediction
testData = np.genfromtxt('20142626-test.csv', delimiter=',').astype(float)
Atest = testData[0:testData.shape[0],0:colT-1]
Atest1 = np.hstack((np.ones((10, 1)), Atest))

#Prediction
Yprediction = np.dot(Atest1, w)
print " Y = ", Yprediction

#Save to file:
result = np.concatenate((Atest, Yprediction), axis = 1)
np.savetxt('20142626.csv', result, delimiter=',')

