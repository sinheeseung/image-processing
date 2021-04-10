import cv2
import numpy as np
# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.my_filtering import my_filtering

def get_DoG_filter(fsize, sigma):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[(fsize - 1) / 2 * -1:fsize / 2, (fsize - 1) / 2 * -1:fsize / 2]
    # x는 -n ~ n 범위의 mask에서 x좌표
    # y는 -n ~ n 범위의 mask에서 y좌표
    # n은 mask의 행 or 열 길이 // 2
    x_2 = np.multiply(x, x)
    y_2 = np.multiply(y, y)
    sig = sigma * sigma
    DoG_x = np.exp(-1 * (x_2 + y_2) / (2*sig)) * (-1 * x / sig)
    #d/dx = -(x/sigma^2) * e^((-x^2+y^2)/ 2*sigma^2)
    DoG_y = np.exp(-1 * (x_2 + y_2) / (2*sig)) * (-1 * y / sig)
    #d/dy = -(y/sigma^2) * e^((-x^2+y^2)/ 2*sigma^2)
    DoG_x /= np.sum(DoG_x)
    DoG_y /= np.sum(DoG_y)
    return DoG_x, DoG_y

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    ###################################################
    # TODO                                            #
    # DoG mask sigma값 조절해서 mask 만들기              #
    ###################################################
    # DoG_x, DoG_y filter 확인
    x, y = get_DoG_filter(fsize=256, sigma=30)
    x = ((x - np.min(x)) / np.max(x - np.min(x)) * 255).astype(np.uint8)
    y = ((y - np.min(y)) / np.max(y - np.min(y)) * 255).astype(np.uint8)
    #가우시안 필터에서 값들이 전부 작아서 잘 안보인다.
    #하지만 출력 이미지에는 잘 나타나게 해주어야 한다.
    #따라서 값을 우선 0~1사이로 바꿔준 다음에 255를 곱해 0~255사이의 값으로 만들어 주었다.
    dst_x = my_filtering(src, DoG_x, 'zero')
    dst_y = my_filtering(src, DoG_y, 'zero')

    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################

    dst_x2 = np.multiply(dst_x,dst_x)
    dst_y2 = np.multiply(dst_y,dst_y)
    dst = np.sqrt(dst_x2 + dst_y2)
    #magnitude = sqrt(I(x)^2 + I(y)^2)

    print(dst)
    cv2.imshow('DoG_x filter', x)
    cv2.imshow('DoG_y filter', y)
    cv2.imshow('dst_x', dst_x/255)
    cv2.imshow('dst_y', dst_y/255)
    cv2.imshow('dst', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

