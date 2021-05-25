import numpy as np
import cv2
import time
def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    ######################################
    h,w = src.shape
    x,y = h,w

    # img가 8*8 block으로 떨어지지 않는 경우, 8의 배수로 맞춰줌
    if(h % n != 0) : x = x + n -(h % n)
    if(w % n != 0) : y = y + n -(w % n)

    pad_img = np.zeros((x,y))
    pad_img[:h,:w] = src.copy()
    #src보다 큰 나머지 부분은 0으로 채워줌(padding)
    blocks = []
    for i in range(x//n):
        #n*n의 block으로 나누어 주기 때문에 *n
        i = i * n
        for j in range(y//n):
            j = j * n
            block = pad_img[i:i+n,j:j+n]
            #이미지를 8*8 block으로 나눠주고 blocks 배열에 저장
            blocks.append(block)
    return np.array(blocks)

def C(w, n=8):
    if w == 0:
        return (1 / n) ** 0.5
    else:
        return (2 / n) ** 0.5

def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    ######################################
    v, u = block.shape
    y, x = np.mgrid[0:u, 0:v]
    mask = np.zeros((n,n))

    for v_ in range(v):
        for u_ in range(u):
         #  F(u,v) = C(u)C(v)*f(x,y)*cos(((2x+1)u*pi)/2n)*cos(((2y+1)v*pi)/2n)
            mask[v_,u_ ] = C(u_, n) * C(v_, n) *np.sum(block * np.cos(((2 * x + 1) * u_ * np.pi) /
                                                            (2 * n)) * np.cos(((2 * y + 1) * v_ * np.pi) / (2 * n)))
    return np.round(mask)

def EOB(dst,zero_index):
    # EOB포함한 배열 생성
    answer = []
    for i in range(zero_index):
        answer.insert(i,dst[i])
    answer.insert(zero_index, 'EOB')
    return answer

def my_zigzag_scanning(block, mode, block_size):
    ######################################
    # TODO                               #
    # my_zigzag_scanning 완성             #
    ######################################
    i,j,count,index =0,0,0,0
    bool_check = False
    if(mode == 'encoding'):
        #인코딩인 경우
        dst = []
        zero_index = -1
        #EOB위치, EOB가 없는 경우 -1로 지정
        while(count < block_size**2):
            # 2차원 배열 => 1차원 배열
            dst.insert(count,block[i,j])
            if(block[i,j] == 0):
                #0인 경우 EOB일 수 있다
                if zero_index == -1:
                    # EOB의 위치가 아직 정해지지 않은 경우
                    zero_index = count
            else:
                # 0이 아닌 숫자가 나오면 EOB가 아니다
                zero_index = -1
            if((i == 0 or j == block_size-1) and bool_check == False):
                #이동 방향이 오른쪽 위이고 다음 방향이
                # 대각선이 아니라 아래나 오른쪽으로 이동하는 경우
                if(j >= block_size-1):
                    # j가 배열의 오른쪽 끝이면 아래로 이동
                    i+=1
                else :
                    # i가 0인 경우, 오른쪽으로 이동
                    j += 1
                bool_check = True
                #다음 이동방향은 왼쪽 아래 방향
            elif((j == 0 or i == block_size-1) and bool_check == True):
                #이동 방향이 왼쪽 아래이고 다음 방향이
                #대각선이 아니라 아래나 오른쪽으로 이동하는 경우
                if(i >= block_size-1):
                    #i가 배열의 맨 아래쪽이면 오른쪽으로 이동
                    j+=1
                else:
                    #j가 0인 경우, 아래로 이동
                    i += 1
                bool_check = False
                #다음 이동 방향은 오른쪽 위 방향
            else:
                #대각선으로 이동하는 경우
                if(not bool_check):
                    i -= 1
                    j += 1
                    #오른쪽 위로 이동 bool_check = false
                else:
                    i += 1
                    j -= 1
                  #왼쪽 아래 이동 bool_check = true
            count += 1
        ans = EOB(dst, zero_index)
        #EOB를 포함한 배열 생성
        return ans
    else:
        dst = np.zeros((block_size,block_size))
        EOB_EXIST = False
        #EOB가 존재하는지 확인하는 변수
        while (count < block_size ** 2):
            if(EOB_EXIST or block[count] == 'EOB'):
                #EOB가 존재하거나 EOB를 마주친 경우 다음에 오는 모든 값들은 0
                EOB_EXIST = True
                value = 0
            else:
                # EOB가 아니면 block에서 값 가져옴
                value = block[count]
            dst[i,j] = value
            if ((i == 0 or j == block_size - 1) and bool_check == False):
                # 이동 방향이 오른쪽 위이고 다음 방향이
                # 대각선이 아니라 아래나 오른쪽으로 이동하는 경우
                if (j >= block_size - 1):
                    # j가 배열의 오른쪽 끝이면 아래로 이동
                    i += 1
                else:
                    # i가 0인 경우, 오른쪽으로 이동
                    j += 1
                bool_check = True
                # 다음 이동방향은 왼쪽 아래 방향
            elif ((j == 0 or i == block_size - 1) and bool_check == True):
                # 이동 방향이 왼쪽 아래이고 다음 방향이
                # 대각선이 아니라 아래나 오른쪽으로 이동하는 경우
                if (i >= block_size - 1):
                    # i가 배열의 맨 아래쪽이면 오른쪽으로 이동
                    j += 1
                else:
                    # j가 0인 경우, 아래로 이동
                    i += 1
                bool_check = False
                # 다음 이동 방향은 오른쪽 위 방향
            else:
                # 대각선으로 이동하는 경우
                if (not bool_check):
                    i -= 1
                    j += 1
                    # 오른쪽 위로 이동 bool_check = false
                else:
                    i += 1
                    j -= 1
                # 왼쪽 아래 이동 bool_check = true
            count += 1
        return dst

def C_inv(w, n=8):
    # idct의 C를 구하는 함수
    # 배열을 인자로 받음
    dst = np.zeros((n,n))
    for x in range(n):
        for y in range(n):
            if(w[x,y] == 0):
                dst[x,y] = (1 / n) ** 0.5
            else:
                dst[x,y] = (2 / n) ** 0.5
    return dst

def DCT_inv(block, n = 8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################
    mask = np.zeros(block.shape)

    v, u = mask.shape
    y, x = np.mgrid[0:u, 0:v]

    C_inv_y = C_inv(y, n=n)
    C_inv_x = C_inv(x, n=n)

    for v_ in range(v):
        for u_ in range(u):
            #f(x,y) = sigma(F(u,v)*C(u)*C(v)*cos(((2x+1)u*pi)/2n)*cos(((2y+1)v*pi)/2n)
            mask[v_,u_] = np.sum(C_inv_x * C_inv_y * block * np.cos(((2 * u_ + 1) *
                            x * np.pi) / (2 * n)) * np.cos(((2 * v_ + 1) * y * np.pi) / (2 * n)))

    mask = np.clip(mask, -128, 127)
    # overflow, underflow
    # 원래는 0 ~ 255이지만 이후 과정 중에 +128을 해주므로 -128 ~ 127

    return np.round(mask)

def block2img(blocks, src_shape, n = 8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    h,w = src_shape
    x,y = h,w
    #img가 8*8 block으로 떨어지지 않은 경우, 8의 배수로 맞춰줌
    if (h % n != 0): x = h + (n - h % n)
    if (w % n != 0): y = w + (n - w % n)
    dst = np.zeros((x,y))
    index = 0
    for i in range(x // n):
        i = i * n
        for j in range(y // n):
            j = j * n
            dst[i: i+n, j: j+n] = blocks[index]
            index += 1
    dst = dst[:h, :w]
    return dst.astype(np.uint8)

def Encoding(src, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)
    #subtract 128
    blocks -= 128
    #DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)
    #print(blocks_dct)
    #Quantization + thresholding
    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q)
    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_scanning(QnT[i], mode ='encoding', block_size=n));
    return zz, src.shape

def Decoding(zigzag, src_shape, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
    blocks = np.array(blocks)
    # Denormalizing
    Q = Quantization_Luminance()
    blocks = blocks * Q

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    # add 128
    blocks_idct += 128
    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst



def main():
    start = time.time()
    #src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    #comp, src_shape = Encoding(src, n=8)

    # 과제의 comp.npy, src_shape.npy를 복구할 때 아래 코드 사용하기(위의 2줄은 주석처리하고, 아래 2줄은 주석 풀기)
    comp = np.load('comp.npy', allow_pickle=True)
    src_shape = np.load('src_shape.npy')

    recover_img = Decoding(comp, src_shape, n=8)
    total_time = time.time() - start

    print('time : ', total_time)
    if total_time > 45:
        print('감점 예정입니다.')
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()