import cv2
from skimage.feature import local_binary_pattern
import numpy as np

#########################
# Adjustable parameters #
#########################

#determines black pixels
threshold = 170

#lbp algo. radius
radius = 4

#no. of points in lbp
noPoints = 8

#########################
#########################


maxVal = 2**noPoints

def lbp(img):
    #convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #call lbp function
    lbp = local_binary_pattern(gray, noPoints, radius, 'default')
    
    #use values corresponding to black pixels only
    npgray = np.asarray(gray)
    y = lbp[np.where(npgray<threshold)]
    
    #construct histogram 
    hist, _ = np.histogram(y, density=True, bins=maxVal, range=(0, maxVal))
    
    #normalize histogram using sum
    sm = sum(hist)
    for i in range (0,maxVal):
      hist[i]=float(hist[i])/sm
    return hist



