from __future__ import print_function

import cv2
import numpy as np
import numpy.polynomial.polynomial as poly

def CIELab_gray(im):
    assert len(im.shape) == 3
    Lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    L, _, _ = cv2.split(Lab)
    return L

def otsu(im):
    _, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # debug_imwrite('otsu.png', thresh)
    return thresh

def premultiply(im):
    assert im.dtype == np.uint8
    im32 = im[:, :, :-1].astype(np.uint32)
    im32 *= im[:, :, -1:]
    im32 >>= 8
    return im32.astype(np.uint8)

def grayscale(im, algorithm=CIELab_gray):
    if len(im.shape) > 2:
        if im.shape[2] == 4:
            return algorithm(premultiply(im))
        else:
            return algorithm(im)
    else:
        return im

def binarize(im,  gray=CIELab_gray, resize=1.0):
    if (im + 1 < 2).all():  # black and white
        
        return im
    else:
        if resize < 0.99 or resize > 1.01:
            
            im = cv2.resize(im, (0, 0), None, resize, resize)
        return otsu(grayscale(im, algorithm=gray))


