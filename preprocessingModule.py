# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:34:49 2018
@author: Reham Abdallatef
"""
from __future__ import division
import cv2



def crop(img):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    immg, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    width=[]
    hight=[]
    rows=[]
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        width.append(w)
        hight.append(h)
        rows.append(y)
        #print(w)
        # Getting ROI
        #roi = img[y:y+h, x:x+w]
    width[width.index(max(width))]=0
    cutupper=0
    cutdown=0
    for i in range(0,3):
        widd=width.index(max(width))
        if rows[widd]<1000:
            if cutupper<rows[widd]:
                cutupper=rows[widd]+hight[widd]
                #roi=img[rows[widd]+hight[widd]:,:]
        else:
            #roi =img[:rows[widd]-hight[widd],:] 
            cutdown = rows[widd]-hight[widd]
        width[width.index(max(width))]=0
    
    return img[cutupper:cutdown,:]