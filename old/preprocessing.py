import cv2
import numpy as np
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
