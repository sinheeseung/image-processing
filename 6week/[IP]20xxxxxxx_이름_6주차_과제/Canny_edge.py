import cv2
import numpy as np
from collections import deque
from my_library.my_filtering import my_filtering
from my_library.my_filtering import my_padding
# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨

    ###########################################
    # TODO                                    #
    # apply_lowNhigh_pass_filter 완성          #
    # Ix와 Iy 구하기                            #
    ###########################################
    y, x = np.mgrid[(fsize - 1) / 2 * -1:fsize / 2, (fsize - 1) / 2 * -1:fsize / 2]
    # x는 -n ~ n 범위의 mask에서 x좌표
    # y는 -n ~ n 범위의 mask에서 y좌표
    # n은 mask의 행 or 열 길이 // 2
    x_2 = np.multiply(x, x)
    y_2 = np.multiply(y, y)
    sig = sigma * sigma

    Ix = np.exp(-1 * (x_2 + y_2) / (2*sig)) * (-1 * x / sig)
    #d/dx = -(x/sigma^2) * e^((-x^2+y^2)/ 2*sigma^2)
    Iy = np.exp(-1 * (x_2 + y_2) / (2*sig)) * (-1 * y / sig)
    #d/dy = -(y/sigma^2) * e^((-x^2+y^2)/ 2*sigma^2)
    Ix = my_filtering(src, Ix, 'zero')
    Iy = my_filtering(src, Iy, 'zero')
    return Ix, Iy

# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    magnitude = np.sqrt(Ix2 + Iy2)
    return magnitude

# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    ###################################################
    # TODO                                            #
    # calcAngle 완성                                   #
    # angle     : ix와 iy의 angle                      #
    # e         : 0으로 나눠지는 경우가 있는 경우 방지용     #
    # np.arctan 사용하기(np.arctan2 사용하지 말기)        #
    ###################################################
    e = 1E-6
    angle = np.arctan(Iy/(Ix+e))
    return angle

def non_maximum_supression(magnitude, angle):
    # angle = -90 ~ +90
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    #제로패딩해서 가장자리 부분도 supression이 가능하게함
    (h, w) = magnitude.shape

    largest_magnitude = np.zeros((h,w))
    for row in range(1,h-1):
        for col in range(1,w-1):
            degree = angle[row,col]

            if 0<= degree and degree < 45:
                rate = np.tan(np.deg2rad(degree))
                left_magnitude = (rate) * magnitude[row-1,col-1] + (1 - rate) * magnitude[row,col-1]
                right_magnitude = (rate) * magnitude[row+1,col+1] + (1 - rate) * magnitude[row,col+1]
                if magnitude[row,col] == max(left_magnitude, magnitude[row,col], right_magnitude):
                    largest_magnitude[row,col] = magnitude[row,col]

            elif 45<= degree and degree <= 90:
                rate = 1/np.tan(np.deg2rad(degree))
                up_magnitude = (1-rate) * magnitude[row-1,col] + rate * magnitude[row-1,col-1]
                down_magnitude = (1-rate) * magnitude[row+1,col] + rate * magnitude[row+1,col+1]
                if magnitude[row,col] == max(up_magnitude, magnitude[row,col], down_magnitude):
                    largest_magnitude[row,col] = magnitude[row,col]

            elif -45<= degree and degree < 0:
                rate = -np.tan(np.deg2rad(degree))
                left_magnitude = (1-rate) * magnitude[row,col-1] + rate * magnitude[row+1,col-1]
                right_magnitude = (1-rate) * magnitude[row,col+1] + rate * magnitude[row-1,col+1]
                if magnitude[row,col] == max(left_magnitude, magnitude[row,col], right_magnitude):
                    largest_magnitude[row,col] = magnitude[row,col]

            elif -90<= degree and degree <-45:
                rate = -1/np.tan(np.deg2rad(degree))
                up_magnitude = (1-rate) * magnitude[row-1,col] + rate * magnitude[row-1,col+1]
                down_magnitude = (1-rate) * magnitude[row+1,col] + rate * magnitude[row+1,col-1]
                if magnitude[row,col] == max(up_magnitude, magnitude[row,col], down_magnitude):
                    largest_magnitude[row,col] = magnitude[row,col]
    return largest_magnitude

def search_weak_edge(dst, edges, high_threshold_value, low_threshold_value):
    (row,col) = edges[-1]
    for i in range(-1,2):
        for j in range(-1,2):
            if dst[row+i,col+j] < high_threshold_value and dst[row+i,col+j] >= low_threshold_value:
                if edges.count((row+i, col+j)) < 1:
                    edges.append((row+i,col+j))
                    search_weak_edge(dst, edges, high_threshold_value,low_threshold_value)

def calssify_edge(dst, weak_edge, high_threshold_value):
    for idx in range(len(weak_edge)):
        (row,col) = weak_edge[idx]
        value = np.max(dst[row-1:row+2,col-1:col+2])
        if value >= high_threshold_value:
            return True

# double_thresholding 수행
def double_thresholding(src):
    dst = src.copy()

    #dst => 0 ~ 255
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)
    (h, w) = dst.shape
    high_threshold_value, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    # high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고
    # low threshold값은 (high threshold * 0.4)로 구한다
    low_threshold_value = high_threshold_value * 0.4
    ######################################################
    # TODO                                               #
    # double_thresholding 완성                            #
    # dst     : double threshold 실행 결과 이미지           #
    ######################################################
    for row in range(h):
        for col in range(w):
            if dst[row,col] >= high_threshold_value:
                dst[row,col] = 255
            elif dst[row,col] < low_threshold_value:
                dst[row,col] = 0
            else:
                weak_edge = []
                weak_edge.append((row,col))
                search_weak_edge(dst, weak_edge, high_threshold_value, low_threshold_value)
                if calssify_edge(dst, weak_edge, high_threshold_value):
                    for idx in range(len(weak_edge)):
                        (r,c) = weak_edge[idx]
                        dst[r,c] = 255
                else:
                    for idx in range(len(weak_edge)):
                        (r,c) = weak_edge[idx]
                        dst[r,c] = 0
    return dst


def my_canny_edge_detection(src, fsize=3, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    # DoG 를 사용하여 1번 filtering
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)


    # Ix와 Iy 시각화를 위해 임시로 Ix_t와 Iy_t 만들기
    Ix_t = np.abs(Ix)
    Iy_t = np.abs(Iy)
    Ix_t = Ix_t / Ix_t.max()
    Iy_t = Iy_t / Iy_t.max()

    cv2.imshow("Ix", Ix_t)
    cv2.imshow("Iy", Iy_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    print(angle)
    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    magnitude_t = magnitude
    magnitude_t = magnitude_t / magnitude_t.max()
    cv2.imshow("magnitude", magnitude_t)

    # non-maximum suppression 수행
    largest_magnitude = non_maximum_supression(magnitude, angle)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    largest_magnitude_t = largest_magnitude
    largest_magnitude_t = largest_magnitude_t / largest_magnitude_t.max()
    cv2.imshow("largest_magnitude", largest_magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # double thresholding 수행
    dst = double_thresholding(largest_magnitude)
    return dst

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original', src)
    dst = my_canny_edge_detection(src)

    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()





''' dst = my_padding(magnitude,(1,1),'zero').astype(np.float64) # size ( h+2, w+2 )
    #magnitude 의 모든 픽셀을 돌며 검사
    h,w = magnitude.shape
    largest_magnitude = np.zeros((h,w))
    for row in range(1,h+1):
        for col in range(1,w+1):
            # 각 픽셀의 엣지와 수직인 각도인 angle을 통해 case 를 나눈다.
            grad = angle[row-1][col-1] # -ㅠ/2 ~ + ㅠ/2
            if grad == 0:
                #수직인 각도가 0도인 경우 좌우값과 비교해주면 된다
                f1 = dst[row][col-1]
                f2 = dst[row][col+1]
            elif abs(grad) == np.pi/2:
                #수작인 각도가 -90, 90도인 경우 위 아래 값과 비교해주면 된다
                f1 = dst[row-1][col]
                f2 = dst[row+1][col]
            elif 0 < grad < np.pi/2:
                #수직인 각도가 0~90도인 경우 좌표상에서 2사분면과 4사분면 방향의 값을 비교해 주면 된다
                # --인 값과 ++인 값을 비교
                dx = abs(np.cos(grad))
                dy = abs(np.sin(grad))
                f1 = (1-dx)*(1-dy)*dst[row][col] + dx*(1-dy)*dst[row][col+1] + \
                     (1-dx)*dy*dst[row+1][col] + dx*dy*dst[row+1][col+1]
                #4사분면 방향
                f2 = dx*dy*dst[row-1][col-1] + (1-dx)*dy*dst[row-1][col] + \
                     dx*(1-dy)*dst[row][col-1] + (1-dx)*(1-dy)*dst[row][col]
                #1사분면 방향
            else:
                #수직인 각도가 -90~0도인 경우 좌표상에서 1사분면과 3사분면 방향의 값을 비교해 주면 된다
                # +-인 값과 -+인 값을 비교
                dx = abs(np.cos(grad))
                dy = abs(np.sin(grad))
                f1 = (1-dx)*dy * dst[row-1][col] + dx*dy*dst[row-1][col+1] + \
                     (1-dx)*(1-dy)*dst[row][col] + dx*(1-dy)*dst[row][col+1]
                #1사분면 방향
                f2 = dx*(1-dy) * dst[row][col-1] + (1-dx)*(1-dy)*dst[row][col] + \
                     dx*dy*dst[row+1][col-1] + (1-dx)*dy*dst[row+1][col]
                #3사분면 방향
            if magnitude[row-1][col-1] >= f1 and magnitude[row-1][col-1] >= f2:
                #magnitude가 주위 값보다 크거나 같은 경우우
               largest_magnitude[row-1][col-1] = magnitude[row-1][col-1]
    return largest_magnitude'''



'''   for i in range(h):
        for j in range(w):
            if dst[i][j] >= int(high_threshold_value):
                dst[i][j] = 255   #강한엣지
            elif int(high_threshold_value) > dst[i][j] > int(low_threshold_value):
                dst[i][j] = 50   #약한엣지
            else:
                dst[i][j] = 0

    #BFS를 통해 약한 엣지 -> 강한 엣지 탐색
    visit = np.zeros((h,w))
    #한 번 방문했던 좌표 기억
    for i in range(h):
        for j in range(w):
            if dst[i][j] == 255 and visit[i][j] == 0:
                #강한 엣지이고 방문한 적 없는 경우
                queue = deque()
                queue.append((i, j))
                while queue: #  BFS 진행
                    x, y = queue.popleft()
                    visit[x][y] = 1
                    #방문 체크
                    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
                    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
                    for k in range(8):
                        nx = x + dx[k]
                        ny = y + dy[k]
                        if nx >= 0 and nx < h and ny >= 0 and ny < w and dst[nx][ny] == 50:
                            # 강한 엣지와 연결된 약한 엣지인 경우
                            queue.append((nx, ny))
                            dst[nx][ny] = 255  # 강한 엣지

    # 나머지 값은 다 0
    for i in range(h):
        for j in range(w):
            if dst[i][j] != 255:
                dst[i][j] = 0

    return dst'''