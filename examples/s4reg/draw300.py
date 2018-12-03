import sys
import numpy as np
import cv2
import copy
import os
import re
import numpy.random as npr
wk_dir = "/Users/momo/wkspace/gesture/data/trainData/T_5_five-pink_300S30/"

if __name__=='__main__':
    pattern = re.compile(r'^[^\.].+\.jpg$')
    numPic = 0
    for dirpath, dirnames, filenames in os.walk(wk_dir):
        for img_filename in filenames:
            match = pattern.match(img_filename)
            if not match: continue
            pt_filename = img_filename.replace('.jpg', '.txt')
            ptfile_fullpath = os.path.join(dirpath, pt_filename)
            label = np.loadtxt(ptfile_fullpath, dtype=float)
            numPic += 1
            print pt_filename, label
            im_small = cv2.imread(os.path.join(dirpath, img_filename))
            show_x1 = int(label[0])
            show_y1 = int(label[1])
            show_x2 = int(label[2])
            show_y2 = int(label[3])
            print show_x1, show_y1, show_x2, show_y2
            cv2.rectangle(im_small,(show_x1, show_y1),(show_x2,show_y2), (255, 0, 0), 2 )
            cv2.imshow("img", im_small)
            cv2.waitKey(1)
    print "the end! %d pic in dir"%numPic