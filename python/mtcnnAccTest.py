import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import scipy.io as sio
import os, sys
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
# Make sure that caffe is on the python path:
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import argparse

import time


if __name__ == '__main__':
    
    caffe.set_mode_cpu()
    inputSize = 48
    mean = 128
    NumTest = 200
    bbox_reg_net = caffe.Net("/Users/momo/Desktop/gesture/fromAli/reg/48net.prototxt", "/Users/momo/Desktop/gesture/fromAli/reg/48net3iter10w.caffemodel", caffe.TEST)

    fid = open("/Users/momo/wkspace/caffe_space/mtcnn-caffe/prepare_data/xhsy48/cls.txt","r")

    lines = fid.readlines()
    fid.close()
    cur_=0
    sum_=len(lines)
    roi_list = []
    clslosssum = 0
    reglosssum = 0
    roi_n = 0
    cls_n = 0
    for line in lines:
        cur_+=1
#        if cur_ == NumTest:
        if not line:
            break;
        words = line.split()
        image_file_name = "/Users/momo/wkspace/caffe_space/mtcnn-caffe/prepare_data/" + words[0] + '.jpg'
#        image_file_name = "/Users/momo/wkspace/caffe_space/mtcnn-caffe/48/0706_7_noscale/" + words[0] + '.jpg'
        print cur_, image_file_name, reglosssum, roi_n
        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
        if h!=inputSize or w!=inputSize:
            im = cv2.resize(im,(int(inputSize),int(inputSize)))
        im = np.swapaxes(im, 0, 2)
        im -= mean
        label    = int(words[1])
        roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
#        roi_list.append([im,label,roi])

        bbox_reg_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        bbox_reg_net.blobs['data'].data[...]=im

#        startT48 = time.clock()
        out_ = bbox_reg_net.forward()
#        endT48 = time.clock()

        prob = out_['prob'][0][1]
        if label != 0:
            roi_n+=1
            box_deltas = out_['fc6-2'][0]
            regloss = np.sum((box_deltas-roi)**2)/2
            reglosssum += regloss
#        print "bbox_reg:",box_deltas
#        print "      gt:",roi_list[cur_-1][2], "loss:", loss, "prob:", out_['prob'][0][1]
    print "average loss:", reglosssum/roi_n

#    fid = open("/Users/momo/wkspace/caffe_space/mtcnn-caffe/prepare_data/xhsy48/cls.txt","r")
##    listfile=open("/Users/momo/wkspace/caffe_space/mtcnn-caffe/prepare_data/py_roi.txt", "w")
#
#    lines = fid.readlines()
#    fid.close()
#    cur_=0
#    sum_=len(lines)
#    roi_list = []
#    losssum = 0
#    for line in lines:
#        cur_+=1
#        if cur_ == NumTest:
#            break;
#        words = line.split()
#        image_file_name = "/Users/momo/wkspace/caffe_space/mtcnn-caffe/prepare_data/" + words[0] + '.jpg'
#        print image_file_name
#        im = cv2.imread(image_file_name)
#
#        h,w,ch = im.shape
#        if h!=inputSize or w!=inputSize:
#            im = cv2.resize(im,(int(inputSize),int(inputSize)))
#        im = np.swapaxes(im, 0, 2)
#
#        im -= mean
#
#        label    = int(words[1])
#        roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
#        roi_list.append([im,label,roi])
#
#        bbox_reg_net.blobs['data'].reshape(1,3,inputSize,inputSize)
#            bbox_reg_net.blobs['data'].data[...]=im
#
#            startT48 = time.clock()
#            out_ = bbox_reg_net.forward()
#            endT48 = time.clock()
#
#            box_deltas = out_['fc6-2'][0]
#            prob = out_['prob'][0][1]
#
#            loss = np.sum((box_deltas-roi_list[cur_-1][2])**2)/2
#            losssum += loss
#            print "bbox_reg:",box_deltas
#            print "      gt:",roi_list[cur_-1][2], "loss:", loss, "prob:", out_['prob'][0][1]
#        print "ave acc:", losssum/NumTest



