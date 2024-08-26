from __future__ import division, print_function

import cv2
import numpy as np
from numpy.polynomial import Polynomial as Poly
from skimage.measure import ransac


class Letter(object):
    def __init__(self, label, label_map, stats, centroid):
        self.label = label
        self.label_map = label_map
        self.stats = stats
        self.centroid = centroid

    @property
    def x(self): return self.stats[cv2.CC_STAT_LEFT]

    @property
    def y(self): return self.stats[cv2.CC_STAT_TOP]

    @property
    def w(self): return self.stats[cv2.CC_STAT_WIDTH]

    @property
    def h(self): return self.stats[cv2.CC_STAT_HEIGHT]

    def area(self):
        return self.stats[cv2.CC_STAT_AREA]

    def __iter__(self):
        return (x for x in self.tuple())

    def tuple(self):
        return (self.x, self.y, self.w, self.h)

    def left(self):
        return self.x

    def right(self):
        return self.x + self.w

    def top(self):
        return self.y

    def bottom(self):
        return self.y + self.h

    def left_mid(self):
        return np.array((self.x, self.y + self.h / 2.0))

    def right_mid(self):
        return np.array((self.x + self.w, self.y + self.h / 2.0))

    def left_bot(self):
        return np.array((self.x, self.y + self.h))

    def right_bot(self):
        return np.array((self.x + self.w, self.y + self.h))

    def corners(self):
        return np.array((
            (self.x, self.y),
            (self.x, self.y + self.h),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h)
        ))

    def base_point(self):
        return np.array((self.x + self.w / 2.0, self.y + self.h))

    def top_point(self):
        return np.array((self.x + self.w / 2.0, self.y))

#     def crop(self):
#         return Crop(self.x, self.y, self.x + self.w, self.y + self.h)

    def slice(self, im):
        return im[self.y:self.y + self.h, self.x:self.x + self.w]

    def raster(self):
        sliced = self.slice(self.label_map)
        return sliced == self.label

    def top_contour(self):
        return self.y + self.raster().argmax(axis=0)

    def bottom_contour(self):
        return self.y + self.h - 1 - self.raster()[::-1].argmax(axis=0)

    def box(self, im, color=(10, 10, 10), thickness=2):
        cv2.rectangle(im, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      color=color, thickness=thickness)

    def __str__(self):
        return 'Letter[{}, {}, {}, {}]'.format(self.x, self.y, self.w, self.h)

    def __repr__(self): return str(self)

    def get_coord(self):
        return (self.x, self.y, self.x+self.w, self.y+self.h)
