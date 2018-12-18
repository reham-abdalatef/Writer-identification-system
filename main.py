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

start = time.time()


#for j in range (1,6):
j = 1
print("=================test case ",j,"======================")
images = [cv2.imread(file) for file in glob.glob("tests/t"+str(j)+"/training/*.png")] 
features=[]
label=[]
trainingSize = len(images)
for i in range(0,trainingSize):
    features.append(lbp(images[i]))
    label.append((i//2)+1)
    
'''
for i in range(0,len(label)):
    plotList(features[i])
    print(label[i])
'''

svclassifier = SVC(kernel='linear')  
svclassifier.fit(Linefeatures, label)
    
    
testimages= [cv2.imread(file) for file in glob.glob("tests/t"+str(j)+"/test/*.png")] 
testSize = len(testimages)
for i in range (0,testSize):
    print(svclassifier.predict([lbp(testimages[i])]))
    
    
end = time.time()
print(end - start)
        
        
        
        
        
        
        
        
        
      
