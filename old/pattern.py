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
from LBP2 import lbp2
from LBP3 import lbp3
from plotting import plotList
import time



#for testId in range (1,6):
testId = 1
start = time.time()
print("=================test case ",testId,"======================")
images = [cv2.imread(file) for file in glob.glob("manywriters-training/*.png")] 
features=[]
label=[]
trainingSize = len(images)
for i in range(0,trainingSize):
    #plotList(lbp2(images[i]))
    #plotList(lbp3(images[i]))
    features.append(lbp2(images[i]))
    label.append((i//2)+1)
    
'''
for i in range(0,len(label)):
    plotList(features[i])
    print(label[i])
'''
    
    
testimages= [cv2.imread(file) for file in glob.glob("manywriters-test/*.png")] 
testSize = len(testimages)
for i in range (0,testSize):
    test = lbp2(testimages[i])
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