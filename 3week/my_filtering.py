import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        pad_img[0:p_h, 0:p_w] = src[0, 0]
        # 왼쪽 위 모서리 부분 padding
        pad_img[p_h+h:p_h+2*h-1, 0:p_w] = src[h-1, 0]
        # 오른쪽 위 모서리 부분 padding
        pad_img[0:p_h, p_w+w:p_w+2*w-1] = src[0, w-1]
        # 왼쪽 아래 모서리 부분 padding
        pad_img[p_h+h:p_h+2*h-1, p_w+w:p_w+2*w-1] = src[h-1, w-1]
        # 오른쪽 아래 모서리 부분 padding

        #up
        for row in range(p_h):
            for col in range(w):
                pad_img[row,p_w+col] = src[0,col]
                # 위쪽 부분 padding을 진행하므로 원본이미지의 첫번째 줄에서
                # col만 이동하면서 padding을 진행하였습니다
        #down
        for row in range(p_h+h,p_h*2+h):
            for col in range(w):
                pad_img[row, col+p_w] = src[h-1, col]
                # 아래쪽 부분 padding을 진행하므로 원본이미지의 마지막 줄에서
                # col만 이동하면서 padding을 진행하였습니다
        #left
        for col in range(p_w):
            for row in range(h):
                pad_img[row+p_h, col] = src[row, 0]
                # 왼쪽 부분 padding을 진행하므로 원본이미지의 첫번째 열에서
                # row만 이동하면서 padding을 진행하였습니다
        #right
        for col in range(w+p_w,p_w*2+w):
            for row in range(h):
                pad_img[row+p_h, col] = src[row, w-1]
                # 오른쪽 부분 padding을 진행하므로 원본이미지의 마지막 열에서
                # row만 이동하면서 padding을 진행하였습니다
        # pad_image의 모서리 부분은 따로 다 padding을 진행하였기 때문에 각 padding과정에서
        # 반복문의 크기만큼만 반목을 진행하였습니다.
    else:
        print('zero padding')

    return pad_img

def my_filtering(src, ftype, fshape, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (fshape[0]//2, fshape[1]//2), pad_type)
    dst = np.zeros((h, w))
    (row, col) = fshape
    mask = np.ones((row,col),dtype = np.float32)
    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                        #
        # 꼭 한줄로 완성할 필요 없음                           #
        ###################################################
        mask = mask / (row*col)
        # average filter는 filter의 모든 값이 같고 다 더했을 경우
        # 합이 1이 되는 filter입니다. 따라서 1로 초기화 된 mask를
        # filter의 총 수로 나누어 주면 average filter가 됩니다.
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                       #
        # 꼭 한줄로 완성할 필요 없음                          #
        ##################################################
        # mask = ???
        mask = mask - mask
        # mask의 값을 0으로 초기화해줌
        mask[int(row/2),int(col/2)] = 2
        # filter의 가운데의 값을 2로 설정
        mask = mask - np.ones((row,col), dtype=np.float32)/(row*col)
        # sharpening filtering의 경우 가운데 값만 2인 filter에서 average
        # filter를 뺀 값이 됩니다. 따라서 위에서 만든 가운데 값만 2인 filter에서
        # average filter를 빼주어 sharpening filtering를 만들었습니다.
        print(mask)

    #########################################################
    # TODO                                                  #
    # dst 완성                                               #
    # dst : filtering 결과 image                             #
    # 꼭 한줄로 완성할 필요 없음                                 #
    #########################################################
    for i in range(h):
        for j in range(w):
            dst[i][j] = np.sum(np.multiply(mask, src_pad[i:i + row,j: j + col]))
            # multiply함수를 통해 각각의 위치에 맞는 원소들끼리의 곱을 구한 후
            # sum을 통해 그 곱들의 합을 구해주었습니다.
            if dst[i][j] > 255:
                # 합을 구하는 과정에서 오버플로우가 발생할 수도 있기 때문에
                # 발생한 경우 255로 값을 지정해 주었습니다.
                dst[i][j] = 255
            elif dst[i][j] < 0:
                # 합을 구하는 과정에서 언더플로우가 발생할 수도 있기 때문에
                # 발생한 경우 0으로 값을 지정해 주었습니다.
                dst[i][j] = 0
    dst = (dst).astype(np.uint8)


    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # repetition padding test
    rep_test = my_padding(src, (20,20), 'repetition')
    zero_test = my_padding(src, (20,20), 'zero')
    # 3x3 filter
   # dst_average = my_filtering(src, 'average', (3,3))
    #dst_sharpening = my_filtering(src, 'sharpening', (3,3))
    #dst_zero = my_filtering(src, 'zero', (3,3))
    #원하는 크기로 설정
    #dst_average = my_filtering(src, 'average', (7,7))
    #dst_sharpening = my_filtering(src, 'sharpening', (7,7))

    # 11x13 filter
    dst_average = my_filtering(src, 'average', (11,13), 'repetition')
    dst_sharpening = my_filtering(src, 'sharpening', (11,13), 'repetition')

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.imshow('zero padding test', zero_test.astype(np.uint8))
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
