# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:36:48 2018

@author: Reham Abdallatef
"""
from __future__ import division
from numpy import*
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


#-------------------------------------------------------
# split to lines
#------------------------------------------------------
def SplitLines(image):
    
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)
    #cv2.waitKey(0)
    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow('second',thresh)
    #cv2.waitKey(0)
    #dilation
    kernel = np.ones((2,100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)
    #cv2.imshow('dilated',img_dilation)
    #cv2.waitKey(0)
    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    countLines=0
    #lines=[]
    for i, ctr in enumerate(sorted_ctrs):
          # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        
        # Getting ROI
        roi = image[y:y+h, x:x+w]
        
        # show ROI
        #cv2.imshow('segment no:'+str(i),roi)
        #cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        
        if h >50 :
            countLines+=1
            cv2.imwrite(str(i)+".png", roi)
            cv2.waitKey(0)
    return countLines
#-----------------------------------------------------------------------------------------------
######################## Get features depend on line hight from f1 to f6 #################################################
#------------------------------------------------------------------------------------------------
def ExtractF1toF6(img):
    ret, imgf = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    projection=[] 
    topline = 0 
    bottomline = 0
    ubberBaseline = 0
    lowerBaseline = 0

    for i in range(height):
        sumblack=0
        for j in range(width):
            sumblack+=imgf[i][j] 
        projection.append(width - sumblack)
    
    for c in range(len(projection)):
        if projection[c] != 0 :
            topline = projection[c]
            break

    for c in range(len(projection)):
        if projection[c] != 0 :
            bottomline = projection[len(projection)-c-1]
            break
        
    rangesTop = []
    rangesBotton = []
    for l in range(projection.index(np.amax(projection))):
        rangesTop.append(projection[l+1] - projection[l])
    
    for l in range(projection.index(np.amax(projection)), len(projection)-1): 
        rangesBotton.append(projection[l] - projection[l+1])
    

    ubberBaseline = np.amax(rangesTop)
    lowerBaseline = np.amax(rangesBotton)

    f1 = abs(topline - ubberBaseline)
    f2 = abs(ubberBaseline - lowerBaseline)
    f3 = abs(lowerBaseline - bottomline)
    f4 = f1/f2
    f5 = f1/f3
    f6 = f2/f3
    return f1,f2,f3,f4,f5,f6
#-------------------------------------------------------------------------------
######################## Get features depend on line weight from f7 to f8 #################################################
#-------------------------------------------------------------------------------
def ExtractF7AndF8(img,f2):
      #cv2.imshow('gray',gray)
    #cv2.waitKey(0)
    ret, imgf = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    transitions=[] 
    max = 0
    currentIndex = 0
    for i in range(height):
        transitionsNo=0
        for j in range(width-1):
            if imgf[i][j] != imgf[i][j+1]:
                transitionsNo+=1
        if max < transitionsNo :
            max = transitionsNo
            currentIndex = i
        transitions.append(transitionsNo)
    distances = []
    for k in range(width-1):
        start = 0
        end = 0
        if imgf[currentIndex][k] == 0 and imgf[currentIndex][k+1] == 1:
            start = k
            for j in range(k,width -1):
                if imgf[currentIndex][j] == 1 and imgf[currentIndex][j+1] == 0:
                    end = j
                    break
        distance = end - start 
        if distance != 0 and start < end:
            distances.append(end-start)
        
    if len(distances) == 0 :
        return (0,0)
    f7 = median(distances)
    f8 = f2 / f7
    return f7,f8
#--------------------------------------------------------------------------------------
#feature 9 and 10
#----------------------------------------------------------------------------------
def ExtractF9AndF10(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)
    #cv2.waitKey(0)
    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow('second',thresh)
    #cv2.waitKey(0)
    XX=np.empty(0)
    YY=np.empty(0)
    p0=0
    p1=0
    p2=0
    p3=0
    past=0
    #dilation
    for x in range(1,50):
        XX = np.append(XX, math.log(x))
        kernel = np.ones(x, np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        n_white_pix = np.sum(img_dilation == 255)
        YY = np.append(YY, math.log(n_white_pix)-math.log(x))
        if x==1:
            p0=[math.log(1),math.log(n_white_pix)-math.log(x)]
        elif x==49:
            p3=[math.log(49),math.log(n_white_pix)-math.log(x)]
            if p2==0 or p1==p2 :
                p2=[p1[0]+(p3[0]- p1[0]),(p3[1]+(p1[1]-p3[1]))]
        elif p1 !=0 and p2==0 and int(math.log(n_white_pix)-math.log(x))!= past :
            p2=[math.log(x),math.log(n_white_pix)-math.log(x)]
        elif p0 != 0 and p1 ==0 and int(math.log(n_white_pix)-math.log(x))!=past:
            p1=[math.log(x),math.log(n_white_pix)-math.log(x)]
        past = int(math.log(n_white_pix)-math.log(x))
    #print (math.log(x),math.log(n_white_pix)-math.log(x))
    #cv2.imshow('dilated',img_dilation)
    #cv2.waitKey(0)
    if p3[0]== p2[0]:
        p2[0]=p3[0]-p1[0]
    if p1[0]==p2[0]:
        p1[0]=p1[0]-0.3
    '''
    plt.scatter(XX, YY)
    plt.plot([p0[0],p1[0]],[p0[1],p1[1]], 'k-')
    plt.plot([p1[0],p2[0]],[p1[1],p2[1]], 'k-')
    plt.plot([p2[0],p3[0]],[p2[1],p3[1]], 'k-')
    plt.show()
    print(p0)
    print(p1)
    print(p2)
    print(p3)
    '''
    f9=(p1[1]-p2[1])/(p1[0]-p2[0])
    f10=(p2[1]-p3[1])/(p2[0]-p3[0])
    
  
    #print(f9)
    #print(f10)
    return f9,f10
#-----------------------------------------------------------
# get f11
#-----------------------------------------------------------
def rotate(img,theta):
    nwimg = img.copy()
    height, width = img.shape
    for i in range(0,height):
        d = int((height-i)*math.tan(theta))
        for j in range(0,width):
            if j+d>=0 and j+d<width:
                nwimg[i][j]=img[i][j+d]
            else:
                nwimg[i][j]=255
    return nwimg
 
def getScore(img):
    threshold = 0.6
    height, width = img.shape
    ret = 0
    for j in range(0,width):
        cnt = 0
        for i in range(0,height):
            if img[i][j]==0:
                cnt +=1
        if cnt/height >=threshold:
            ret +=1
    return ret
 
def getSlant(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turn RGB to GRAY
    height, width = img.shape    
 
    threshold = 170
    for i in range(0,height):                   # turn GRAY to BLACK and WHITE
        for j in range (0,width):
            if img[i][j] >= threshold:
                img[i][j] = 255
            else : 
                img[i][j] = 0
 
    
    bstScore = getScore(img)
    bstTheta = 0
    theta = -45
    while theta <= 45 :
        tempScore = getScore(rotate(img,theta*math.pi/180))
        if tempScore > bstScore:
            bstScore = tempScore
            bstTheta = theta
        theta += 3
    return bstTheta
def ExtractF11(img):
    f11=getSlant(img)
    print(f11)
    return f11
    
def WIUTL(img):
    countLines=SplitLines(img)
    Linefeatures=[]
    for counter1 in range(0,countLines):
        line = cv2.imread(str(counter1)+".png",0)
        #print("line"+str(x)+".png")
        #cv2.imshow('dilated',line)
        #cv2.waitKey(0)
        (f1,f2,f3,f4,f5,f6)=ExtractF1toF6(line)
        (f7,f8)=ExtractF7AndF8(line,f2)
        line = cv2.imread(str(counter1)+".png")
        (f9,f10)= ExtractF9AndF10(line)
        Linefeatures.append([f5,f6,f8,f9])
    return Linefeatures
#------------------------ END OF Functions -----------------------------------------------
       
 
