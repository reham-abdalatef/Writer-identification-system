import cv2
def lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = [0] * 256
    height, width = gray.shape
    #    7 0 1
    #    6 C 2
    #    5 4 3
    threshold = 170
    for i in range(1, height-1):
        for j in range(1, height-1):
            if gray[i][j] < threshold:
                continue
            val = 0
            if gray[i-1][j] <= gray[i][j]:
                val += 1
            if gray[i-1][j+1] <= gray[i][j]:
                val += 2
            if gray[i][j+1] <= gray[i][j]:
                val += 4
            if gray[i+1][j+1] <= gray[i][j]:
                val += 8
            if gray[i+1][j] <= gray[i][j]:
                val += 16
            if gray[i+1][j-1] <= gray[i][j]:
                val += 32
            if gray[i][j-1] <= gray[i][j]:
                val += 64
            if gray[i-1][j-1] <= gray[i][j]:
                val += 128

            hist[val] += 1

    sm = sum(hist)
    return hist / sm

