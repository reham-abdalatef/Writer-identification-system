from featuresModule import lbp
import numpy as np

def test(img,features,label):
    
    #compute feature vector of test image
    test = lbp(img)
    
    #find nearest neighbour
    trainingSize = len(features)
    bestDist = np.linalg.norm(np.asarray(test) - np.asarray(features[0]))
    bestPredict = label[0]
    for j in range(1,trainingSize):
        curDist = np.linalg.norm(np.asarray(test)-np.asarray(features[j]))
        if curDist < bestDist:
            bestDist = curDist.copy()
            bestPredict = label[j]
    return bestPredict
        
    
    