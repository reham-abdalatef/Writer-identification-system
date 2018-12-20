import cv2
from featuresModule import lbp
from preprocessingModule import crop
import glob

def train(folderName):
    
    #initialize lists
    features = []
    label = []
    
    #iterate over all 3 writers and add their feature vectors
    for i in range(1,4):
        images = [cv2.imread(file) for file in glob.glob("data/"+folderName+ "/" +str(i)+"/*.PNG")]
        for img in images:
            features.append(lbp(crop(img)))
            label.append(i)
            
    return features, label
