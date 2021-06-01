import numpy as np
import cv2

def forward(src, M, fit):
    #####################################################
    # TODO                                              #
    # forward 완성                                      #
    #####################################################
    print('< forward >')
    print('M')
    print(M)
    h, w = src.shape
    dst = np.zeros((src.shape))
    N = np.zeros(dst.shape)
    if (fit):
        for row in range(h):
            for col in range(w):
                P = np.array([[col],
                                [row],
                                [1]])

                P_dst = np.dot(M, P)
                dst_col = P_dst[0][0]
                dst_row = P_dst[1][0]

                dst_col_left = int(dst_col)
                #좌측
                dst_col_right = int(np.ceil(dst_col))
                #우측

                dst_row_top = int(dst_row)
                #상단
                dst_row_bottom = int(np.ceil(dst_row))
                #하단

                if dst_row_top < h and dst_col_left < w:
                    #좌측 상단부터 확인, 계산한 좌표가 범위 안에 있다면
                    dst[dst_row_top, dst_col_left] += src[row, col]
                    #값을 더해주고
                    N[dst_row_top, dst_col_left] += 1
                    #count해준다

                    if dst_col_right != dst_col_left and dst_col_right < w:
                        #dst_col이 정수값이 아니라 왼쪽, 오른쪽 모두에 값이 들어가야 하는 경우
                        #좌표가 범위 안에 들어가는지 확인, 우측 상단
                        dst[dst_row_top, dst_col_right] += src[row, col]
                        #값을 더해주고
                        N[dst_row_top, dst_col_right] += 1
                        #count해준다

                    if dst_row_top != dst_row_bottom and dst_row_bottom < h:
                        # dst_row가 정수값이 아니라 2좌표에 값이 들어가야 하는 경우
                        # 좌표가 범위 안에 들어가는지 확인, 좌측 하단
                        dst[dst_row_bottom, dst_col_left] += src[row, col]
                        #값을 더해주고
                        N[dst_row_bottom, dst_col_left] += 1
                        #count

                    if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top and dst_col_right < w and dst_row_bottom < h:
                        #dst_row, dst_col모두 정수값이 아닌 경우
                        #우측 하단
                        dst[dst_row_bottom, dst_col_right] += src[row,col]
                        #값을 더해주고
                        N[dst_row_bottom, dst_col_right] += 1
                        #count

        dst = np.round(dst / (N + 1E-6))
        dst = dst.astype(np.uint8)
        return dst


def backward(src, M, fit):
    #####################################################
    # TODO                                              #
    # backward 완성                                      #
    #####################################################
    print('< backward >')
    print('M')
    print(M)
    dst = np.zeros(src.shape)
    h, w = dst.shape
    h_src, w_src = src.shape
    M_inv = np.linalg.inv(M)
    print('M inv')
    print(M_inv)

    if (fit):
        for row in range(h):
            for col in range(w):
                P_dst = np.array([
                    [col],
                    [row],
                    [1]
                ])
                P = np.dot(M_inv, P_dst)
                src_col = P[0][0]
                src_row = P[1][0]

                src_col_right = int(np.ceil(src_col))
                src_col_left = int(src_col)
                src_row_bottom = int(np.ceil(src_row))
                src_row_top = int(src_row)

                if src_col_right >= w_src or src_row_bottom >= h_src:
                    #계산한 값이 범위안에 있는지 확인
                    continue

                s = src_col - src_col_left
                t = src_row - src_row_top
                #bilinear interpolation 실행
                intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left] \
                        + s * (1 - t) * src[src_row_top, src_col_right] \
                        + (1 - s) * t * src[src_row_bottom, src_col_left] \
                        + s * t * src[src_row_bottom, src_col_right]
                dst[row, col] = intensity
        dst = dst.astype(np.uint8)
        return dst



def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    #####################################################
    # TODO                                              #
    # M 완성                                             #
    # M_tr, M_sc ... 등등 모든 행렬 M 완성하기              #
    #####################################################
    # translation
    M_tr = np.array([[1, 0, -30],
                     [0, 1, +50],
                     [0, 0, 1]
                     ])

    # scaling
    M_sc = np.array([[0.5, 0, 0],
                     [0, 0.5, 0],
                     [0, 0, 1]
                     ])

    # rotation
    degree = -20
    M_ro = np.array([[np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
                     [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
                     [0, 0, 1]
                     ])

    # shearing
    M_sh = np.array([[1, 0.2, 0],
                     [0.2, 1, 0],
                     [0, 0, 1]
                     ])

    M = np.dot(M_sh, np.dot(M_sc, np.dot(M_tr, M_ro)))
    # fit이 True인 경우와 False인 경우 다 해야 함.
    fit = True
    # forward
    dst_for = forward(src, M, fit=fit)
    dst_for2 = forward(dst_for, np.linalg.inv(M), fit=fit)

    # backward
    dst_back = backward(src, M, fit=fit)
    dst_back2 = backward(dst_back, np.linalg.inv(M), fit=fit)
    fit = False
    cv2.imshow('original', src)
    cv2.imshow('forward2', dst_for2)
    #cv2.imshow('forward1', forward(dst_for, np.linalg.inv(M), fit=fit))
    cv2.imshow('backward2', dst_back2)
    #cv2.imshow('backward1', backward(dst_back, np.linalg.inv(M), fit=fit))
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()