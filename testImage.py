from LBP2 import lbp2
import numpy as np
def testImage(img,features,label):
    test = lbp2(img)
    trainingSize = len(features)
    bstDist = np.linalg.norm(np.asarray(test) - np.asarray(features[0]))
    bstPredict = label[0]
    for j in range(1,trainingSize):
        nwDist = np.linalg.norm(np.asarray(test)-np.asarray(features[j]))
        if nwDist < bstDist:
            bstDist = nwDist.copy()
            bstPredict = label[j]
    return bstPredict
        
    
    