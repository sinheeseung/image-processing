import cv2
import numpy as np

def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                           #
    ###############################################
    (h, w) = B.shape
    (s_h, s_w) = S.shape
    dst = np.zeros((h,w))
    h_2 = int(s_h/2)
    w_2 = int(s_w/2)
    for i in range(h):
        for j in range(w):
            if B[i][j] == 1:  # 흰색일 경우
                k = 0
                for row in range(i - h_2, i + h_2 + 1):
                    l = 0
                    for col in range(j - w_2, j + w_2 + 1):
                        if row >= 0 and row < h and col >= 0 and col < w:
                            # 이미지의 범위 안인 경우
                            dst[row][col] = S[k][l]
                        l += 1
                    k += 1

    return dst

def erosion(B, S):
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                            #
    ###############################################
    (h, w) = B.shape
    (s_h, s_w) = S.shape
    dst = np.zeros((h, w))
    h_2 = int(s_h/2)
    w_2 = int(s_w/2)
    for i in range(h):
        for j in range(w):
            if B[i][j] == 1:  # 흰색일 경우
                cnt = s_h*s_w; k = 0
                #cnt : S와 B의 좌표값이 같은지 확인하는 counter
                for row in range(i - h_2, i + h_2 + 1):
                    l = 0
                    for col in range(j - w_2, j + w_2 + 1):
                        if (row < 0 or row >= h) or (col < 0 or col >= w) or (B[row][col] != S[k][l]):
                            # 이미지의 범위를 벗어나거나 S의 값과 B의 값이 다른 경우
                            cnt -= 1
                            break
                        l += 1
                    if cnt != s_h*s_w :
                        #하나라도 다른 부분이 있으면 그 좌표는 erosion실행 불가
                        break
                    k += 1
                if cnt ==s_h*s_w:
                    dst[i][j] = 1
    return dst

def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                            #
    ###############################################

    #erosion 실행 후 dilation실행
    dst = dilation(erosion(B,S), S)
    return dst

def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                            #
    ###############################################

    #dilation 실행 후 erosion실행
    dst = erosion(dilation(B,S), S)
    return dst



if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('morphology_B.png', (B*255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)


