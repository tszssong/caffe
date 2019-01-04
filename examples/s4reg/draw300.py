import sys
import numpy as np
import cv2
import copy
import os
import re
import numpy.random as npr
wk_dir = "../../data/trainData/T_5_five-pink_300S30/"
wk_dir = "/Users/momo/wkspace/gesture/handreg/data/gt_test_320S32/"
data_dir_path = wk_dir + "testData/"
data_dir_path = wk_dir
model_dir_path = wk_dir + "testData_model/"
TestModelAnno = False
if __name__=='__main__':
    pattern = re.compile(r'^[^\.].+\.jpg$')
    numPic = 0
    for dirpath, dirnames, filenames in os.walk(data_dir_path):
        for img_filename in filenames:
            match = pattern.match(img_filename)
            if not match: continue
            pt_filename = img_filename.replace('.jpg', '.txt')
            if TestModelAnno:
                ptfile_fullpath = os.path.join(model_dir_path, pt_filename)
                showColor = (0, 255, 255)
            else:
                ptfile_fullpath = os.path.join(dirpath, pt_filename)
                showColor = (0, 255, 0)
            label = np.loadtxt(ptfile_fullpath, dtype=float)
            numPic += 1
            print pt_filename, label
            im_small = cv2.imread(os.path.join(dirpath, img_filename))
            show_x1 = int(label[0])
            show_y1 = int(label[1])
            show_x2 = int(label[2])
            show_y2 = int(label[3])
            print show_x1, show_y1, show_x2, show_y2
            cv2.rectangle(im_small,(show_x1, show_y1),(show_x2,show_y2), showColor, 2 )
            cv2.imshow("img", im_small)
            cv2.waitKey(3)
    print "the end! %d pic in dir"%numPic