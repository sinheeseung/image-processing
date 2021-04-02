import numpy as np
import matplotlib.pyplot as plt
import cv2
"""
def my_calc(img):
    h, w = img.shape[:2]
    hist = np.zeros((10,), dtype=np.int)
    for row in range(h):
        for col in range(w):
            intensity = img[row,col]
            hist[intensity] += 1
    return hist

src = np.array([[3,1,3,5,4],[9,8,3,5,6],[2,2,3,8,7],
                [5,4,6,5,4],[1,0,0,2,6]],dtype=np.uint8)
src_visible = (src/9*255).astype(np.uint8)

hist = my_calc(src)
binX = np.arange(len(hist))
plt.bar(binX, hist, width = 0.8, color = 'g')
plt.title('histogram')
plt.xlabel('pixel intensity')
plt.ylabel('pixel num')
plt.show()
"""
src = cv2.imread('fruits.jpg')
dst2 = (src[:,:,0] * 1/3) + (src[:,:,1] * 1/3) + (src[:,:,2]*1/3)
cv2.imshow('dst',dst2)
cv2.waitKey()