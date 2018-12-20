import cv2
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from plotting import plotList
import numpy as np
import time

#########################
# Adjustable parameters #
threshold = 170
radius = 4
noPoints = 8
maxVal = 2**noPoints
#########################

def lbp2(img):
    #start = time.time()
    METHOD = 'default'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, noPoints, radius, METHOD)
    npgray = np.asarray(gray)
    y = lbp[np.where(npgray<threshold)]
    #print(y)
    hist, _ = np.histogram(y, density=True, bins=maxVal, range=(0, maxVal))
    sm = sum(hist)
    for i in range (0,maxVal):
      hist[i]=float(hist[i])/sm
    #plotList(hist)
    #print(time.time()-start)
    return hist