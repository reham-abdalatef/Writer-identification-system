import numpy as np
import cv2
import glob
from cropping import croping
import time

for name in range(0,100):
    folderName = str(name)
    while len(folderName)<3:
        folderName = "0" + folderName
    images = [cv2.imread(file) for file in glob.glob("new_form/"+folderName+ "/*.PNG")]
    for img in images:
        start = time.time()
        cropped = croping(img)
        print(time.time()-start)
        img3 = np.concatenate((img, cropped), axis=0)
        cv2.imshow('image',cv2.resize(img3,(500,980)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

'''
x = np.array([[1,2,3],[3,2,1]])
indices = np.where(x>1)
print(indices)
y = x[indices]

print(y)
'''