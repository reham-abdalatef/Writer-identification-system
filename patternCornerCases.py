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
from plotting import plotList
import time
import random
from testImage import testImage
from cropping import croping

folders = [['087', '151', '352'],['550', '062', '348'],['300', '544', '338'],['544', '243', '239'],['246', '634', '293'],['390', '291', '113'],['272', '671', '348'],['155', '455', '348']]
expected = [1,3,3,1,3,3,3,3]
correct = 0
#random.seed(102) # to make it generate the same tests every time!

for testId in range (0,len(folders)):
    start = time.time()
    print("=================test case ",testId+1,"======================")
    features=[]
    label=[]
    test = []
    for i in range(0,3):
        images = [cv2.imread(file) for file in glob.glob("new_form/"+folders[testId][i]+ "/*.PNG")]
        for j in range(0,2):
            features.append(lbp2(croping(images[j])))
            label.append(i+1)
        if i+1 == expected[testId]:
            test = images[2]
    
    predict = testImage(croping(test),features,label)
    print("expected = " +str(expected[testId]) + " predicted = " + str(predict) )
    if predict == expected[testId]:
        correct += 1 
    if predict != expected[testId]:
        print(folders[testId])
    end = time.time()
    print(end - start)
    
accuracy = 100.0*correct/len(folders)
print("accuracy = " +str(accuracy) + "%")
    
    
    
    
    
    
    
    
    
    
    
    