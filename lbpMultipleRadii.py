import cv2
from skimage.feature import local_binary_pattern
import numpy as np

#########################
# Adjustable parameters #
#########################

#determines black pixels
threshold = 170

#lbp algo. list of radii
radii = [1,5]

#no. of points in lbp
noPoints = 8

#########################
#########################


maxVal = 2**noPoints

def lbp(img):
    #convert image to grayscale
    gray = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    #indices corresponding to black pixels only
    indices = np.where(gray<threshold)
    
    # create empty histogram and append to it for each radius
    hist = np.array([])
    
    for radius in radii:
        
        #call lbp function for current radius
        lbp = local_binary_pattern(gray, noPoints, radius, 'default')
    
        #calculate corresponding histogram
        newHist, _ = np.histogram(lbp[indices], density=True, bins=maxVal, range=(0, maxVal))
            
        #normalize histogram
        sm = np.sum(newHist)
        newHist = np.true_divide(newHist,sm)
        
        #concatenate with result
        hist = np.concatenate((hist, newHist), axis=0)
    
    return hist



