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
from cropping import croping
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
        trainingSize = len(images)
        for img in images:
            features.append(lbp(croping(img)))
            label.append(i)
        
    '''
    for i in range(0,len(label)):
        plotList(features[i])
        print(label[i])
    '''
        
    trainingSize=len(features)    
    testimages= [cv2.imread(file) for file in glob.glob("data/"+ folderName +"/test.PNG")] 
    testSize = len(testimages)
    for i in range (0,testSize):
        test = lbp(croping(testimages[i]))
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
    
    
    
    
