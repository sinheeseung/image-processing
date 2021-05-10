import numpy as np

def my_padding(src, pad_shape, pad_type):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = src
    if pad_type == 'zero':
        print('zero padding')
        pad_img[:p_h, p_w:p_w+w] = 0
        pad_img[p_h+h:, p_w:p_w+w] = 0
        pad_img[:, :p_w] = 0
        pad_img[:, p_w+w:] = 0
    if pad_type == 'repetition':
        print('repetition padding')

        pad_img[:p_h, p_w:p_w+w] = src[0,:]
        pad_img[p_h+h:, p_w:p_w+w] = src[h-1,:]
        pad_img[:, :p_w] = pad_img[:,p_w:p_w+1]
        pad_img[:, p_w+w:] = pad_img[:, p_w+w-1:p_w+w]
    return pad_img

def my_filtering(src, filter, pad_type='zero'):
    (h,w) = src.shape
    (f_h, f_w) = filter.shape
    src_pad = my_padding(src, (f_h//2,f_w//2), pad_type)
    dst = np.zeros((h,w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row+f_h, col:col+f_w] * filter)
            dst[row,col] = val
    return dst