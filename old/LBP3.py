import cv2
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from plotting import plotList
import numpy as np
def lbp3(img):
    threshold = 170
    METHOD = 'default'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, METHOD)
    npgray = np.asarray(gray)
    y = lbp[np.where(npgray<threshold)]
    hist, _ = np.histogram(y, density=True, bins=256, range=(0, 256))
    
    return hist