import numpy as np
import matplotlib.pyplot as plt
import cv2

def my_calc(img):
    h, w = img.shape[:2]
    hist = np.zeros((256,), dtype=np.int)
    for row in range(h):
        for col in range(w):
            intensity = img[row,col]
            hist[intensity] += 1
    return hist

def hist_stretch(src, hist):
    (h,w) = src.shape
    dst = np.zeros((h,w), dtype = np.uint8)
    min = 256
    max = -1

    for i in range(len(hist)):
        if hist[i] != 0 and i < min:
            min = i
        if hist[i] != 0 and i > max:
            max = i

    stretch = np.zeros(hist.shape, dtype = np.int8)
    for i in range(min, max+1):
        j = int((255-0)/(max-min) * (i-min)+0)
        stretch[j] = hist[i]

    for row in range(h):
        for col in range(w):
            dst[row,col] = (255-0)/(max-min) * (src[row,col] -min) + 0

    return dst, stretch

if __name__ == '__main__':
    src = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)
    src_div = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calc(src)
    hist_div = my_calc(src_div)

    dst, stretch = hist_stretch(src_div,hist_div)

    binX = np.arange(len(stretch))
    plt.bar(binX, hist_div, width=0.5, color='g')
    plt.title('divide 3 image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    plt.bar(binX, stretch, width=0.5, color='g')
    plt.title('strecthing image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    plt.bar(binX, hist, width=0.5, color='g')
    plt.title('original image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    cv2.imshow('div 1/3 image', src_div)
    cv2.imshow('stretched image', dst)
    cv2.imshow('original image', src)
    cv2.waitKey()
    cv2.destroyAllWindows()