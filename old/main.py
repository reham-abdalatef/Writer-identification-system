# -*- coding: utf-8 -*-
"""
Created on Wed Dec 05 13:52:42 2018
@author: Reham Abdallatef
"""
from __future__ import division
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier
from numpy import*
import cv2
import numpy as np
import math
from OldFeatures import WIUTL
from cropping import croping
import time
import glob

#----------------------- MAIN ------------------------------------------------------------

for testId in range (1,51):
    start = time.time()
    print("=================test case ",testId,"======================") 
    print("training NOW")
    features=[]
    label=[]
    folderName = str(testId);
    if len(folderName)==1:
        folderName = "0" + folderName
    for i in range(1,4):
        images = [cv2.imread(file) for file in glob.glob("data/"+folderName+ "/" +str(i)+"/*.PNG")]
        trainingSize = len(images)
        for img in images:
            featuresLines=WIUTL(croping(img))
            for j in range(0,len(featuresLines)):
                features.append(featuresLines[j])
                label.append(i)
    #print(features)
    #print(label)       
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(features, label) 
    #svclassifier = SVC(kernel='linear')  
    #svclassifier.fit(Linefeatures, label)
    #Linefeatures = np.isnan(Linefeatures)
    #clf = svm.SVC(kernel='linear')
    #clf.fit(Linefeatures, label)
    #testing
    print("testing NOW")
    count=[0,0,0]
    testimages= [cv2.imread(file) for file in glob.glob("data/"+ folderName +"/test.PNG")] 
    testSize = len(testimages)
    for j in range (0,testSize):
        test = WIUTL(croping(testimages[j]))
        for i in range(0,len(test)):
            predicted= neigh.predict([test[i]])
            if(predicted[0]==1):
                count[0]=count[0] + 1
            elif(predicted[0]==2):
                count[1]=count[1] + 1
            else:
                count[2]= count[2] + 1
        print(count.index(max(count))+1)
        
        
    end = time.time()
    print(end - start) 