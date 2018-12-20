import cv2
import numpy as np
import plotting
from LBP import lbp
import time


image = cv2.imread(r"C:\Users\OmarHashim\Desktop\1.png")

start = time.time()
lbp = lbp(image)
end = time.time()

plotting.plotList(lbp)

print(end - start)