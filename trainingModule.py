import cv2
from featuresModule import lbp
import glob

def train(folderName):
    
    #initialize lists
    features = []
    label = []
    
    #iterate over all 3 writers and add their feature vectors
    for i in range(1,4):
        images = [cv2.imread(file) for file in glob.glob("data/"+folderName+ "/" +str(i)+"/*.PNG")]
        for img in images:
            features.append(lbp(img[900:2860, 20:2479]))
            label.append(i)
            
    return features, label