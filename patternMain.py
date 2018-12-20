from __future__ import division
import cv2
import time
from plotting import plotList
from trainingModule import train
from testingModule import test
from preprocessingModule import crop

##########################
# Adjustable parameters ##
##########################

#no. of tests required to run
testCount = 10

#i.e. '01', '001'
folderNameLen = 2 

##########################
##########################


#clears output files
open('results.txt', 'w').close()
open('time.txt', 'w').close()

for testId in range (1,testCount + 1):
    start = time.time()
    print("running test case #",testId) 
          
    #compute exact folder name
    folderName = str(testId);
    while len(folderName)<folderNameLen:
        folderName = "0" + folderName
        
    #call the training module
    features, label = train(folderName)
    
    #call the testing module
    testImage = cv2.imread("data/"+ folderName +"/test.PNG")
    predict = test(crop(testImage),features,label)
    
    #print prediction and time
    print(predict,file=open("results.txt", "a"))
    end = time.time()
    print("%.2f" %(end - start),file=open("time.txt", "a"))
    
    
    
    
    
    
    
    
    
    
    
    
