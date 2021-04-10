import numpy as np
import cv2
import time

# library add
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.my_filtering import my_padding


def my_get_Gaussian2D_mask(msize, sigma):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    y, x = np.mgrid[(msize-1)/2*-1:msize/2, (msize-1)/2*-1:msize/2]
    #x는 -n ~ n 범위의 mask에서 x좌표
    #y는 -n ~ n 범위의 mask에서 y좌표
    #n은 mask의 행 or 열 길이 // 2
    # 2차 gaussian mask 생성
    x_2 = np.multiply(x, x)
    y_2 = np.multiply(y, y)
    sig = 2 * sigma * sigma
    gaus2D = np.exp(-1*(x_2+y_2)/sig)
    #gaus2D = e^((-x^2+y^2)/ 2*sigma^2)

    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)
    #mask의 총 합을 1로 해주기 위해 전부 다 더한 값으로 나누어 줌

    return gaus2D


def my_get_Gaussian1D_mask(msize, sigma):
    #########################################
    # ToDo
    # 1D gaussian filter 만들기
    #########################################
    x = np.mgrid[(msize-1)/2*-1:msize/2,]
    #x는 -n ~ n 범위의 mask에서 x좌표
    #사용하기 쉽게 2차원 배열로 만듬

    x_2 = np.multiply(x,x)
    sig = 2 *sigma * sigma
    gaus1D = np.exp(-1*(x_2)/sig)
    #gaus2D = e ^ ((-(x ^ 2)) / 2 * sigma ^ 2)

    # mask의 총 합 = 1
    gaus1D /= np.sum(gaus1D)
    #mask의 총 합을 1로 해주기 위해 전부 다 더한 값으로 나누어 줌

    return gaus1D


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    # mask의 크기
    (m_h, m_w) = mask.shape
    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, (m_h // 2, m_w // 2), pad_type)

    print('<mask>')
    print(mask)

    # 시간을 측정할 때 만 이 코드를 사용하고 시간측정 안하고 filtering을 할 때에는
    # 4중 for문으로 할 경우 시간이 많이 걸리기 때문에 2중 for문으로 사용하기.
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            sum = 0
            for m_row in range(m_h):
                for m_col in range(m_w):
                    sum += pad_img[row + m_row, col + m_col] * mask[m_row, m_col]
            dst[row, col] = sum

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    mask_size = 7
    gaus2D = my_get_Gaussian2D_mask(mask_size, sigma=3)
    gaus1D = my_get_Gaussian1D_mask(mask_size, sigma=3)

    print('mask size : ', mask_size)
    print('1D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus1D = my_filtering(src, gaus1D.T)

    dst_gaus1D = my_filtering(dst_gaus1D, gaus1D)
    end = time.perf_counter()  # 시간 측정 끝
    print('1D time : ', end - start)


    print('2D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus2D = my_filtering(src, gaus2D)
    end = time.perf_counter()  # 시간 측정 끝

    print('2D time : ', end - start)
    dst_gaus1D = np.clip(dst_gaus1D + 0.5, 0, 255)
    dst_gaus1D = dst_gaus1D.astype(np.uint8)
    dst_gaus2D = np.clip(dst_gaus2D + 0.5, 0, 255)
    dst_gaus2D = dst_gaus2D.astype(np.uint8)


    cv2.imshow('original', src)
    cv2.imshow('1D gaussian img', dst_gaus1D)
    cv2.imshow('2D gaussian img', dst_gaus2D)
    cv2.waitKey()
    cv2.destroyAllWindows()