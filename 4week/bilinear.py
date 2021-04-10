import cv2
import numpy as np

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)
    dst = np.zeros((h_dst, w_dst))
    # bilinear interpolation 적용
    for row in range(h_dst):
        #512
        for col in range(w_dst):
            px = int(col / scale)
            py = int(row / scale)

            p_px = px + 1
            p_py = py + 1
            # 1~256
            if (p_px >= h): p_px = p_px - 1
            if (p_py >= w): p_py = p_py - 1
            #값을 나누거나 곱하는 과정이기 때문에 0보다 작은값은
            #생기지 않음. 따라서 최대범위를 넘어가는 경우만 예외처리

            p1 = src[py][px]
            p2 = src[py][p_px]
            p3 = src[p_py][px]
            p4 = src[p_py][p_px]
            #값을 구하고자 하는 점과 인접한 4개의 점을 구함

            fx1 = float(col)/ scale - float(px)
            fx2 = 1 - fx1
            fy1 = float(row)/ scale - float(py)
            fy2 = 1 - fy1
            #4개의 점으로부터 거리비를 구한다.

            w1 = fx2*fy2
            w2 = fx1*fy2
            w3 = fx2*fy1
            w4 = fx1*fy1
            #4개의 점와 구하고자 하는 점으로 나누어지는
            #4개 사각형의 크기를 구함


            dst[row][col] = w1*p1+w2*p2+w3*p3+w4*p4
    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 1/7
    #이미지 크기 1/2배로 변경
    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 2배로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


