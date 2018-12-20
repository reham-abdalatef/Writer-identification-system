from __future__ import division
from numpy import*
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
from sklearn import svm
from sklearn.svm import SVC  
from LBP import lbp
from plotting import plotList
import time



for testId in range (1,11):
    start = time.time()
    print("=================test case ",testId,"======================") 
    features=[]
    label=[]
    folderName = str(testId);
    if len(folderName)==1:
        folderName = "0" + folderName
    for i in range(1,4):
        images = [cv2.imread(file) for file in glob.glob("data/"+folderName+ "/" +str(i)+"/*.PNG")]
        lenImages = len(images)
        lbpVec = [0] * 256
        for j in range(0,lenImages):
            nwlbpVec = lbp(images[j][900:2860, 20:2479])
            for k in range(0,256):
                lbpVec[k] +=nwlbpVec[k]
        for j in range(0,256):
            lbpVec[k] /= lenImages
        features.append(lbpVec)
        label.append(i)
    
    '''
    for i in range(0,len(label)):
        plotList(features[i])
        print(label[i])
    '''
    
    trainingSize = len(features)
    testimages= [cv2.imread(file) for file in glob.glob("data/"+ folderName +"/test.PNG")] 
    testSize = len(testimages)
    for i in range (0,testSize):
        test = lbp(testimages[i][900:2860, 20:2479])
        bstDist = np.linalg.norm(np.asarray(test) - np.asarray(features[0]))
        bstPredict = label[0]
        for j in range(1,trainingSize):
            nwDist = np.linalg.norm(np.asarray(test)-np.asarray(features[j]))
            if nwDist < bstDist:
                bstDist = nwDist.copy()
                bstPredict = label[j]
        print(bstPredict)
        
        
    end = time.time()
    print(end - start)
    
    
    
    
    
    
    
    
    
    
    
    