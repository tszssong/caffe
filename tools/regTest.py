# -*- coding: utf-8 -*-
#!/usr/bin/env python
import sys
caffe_root = '/nfs/zhengmeisong/wkspace/gesture/caffe/distribute/'
sys.path.append(caffe_root + 'python')
sys.path.append(caffe_root + 'lib')
#sys.path.append('//nfs/zhengmeisong/wkspace/gesture/caffe/build/python')
#sys.path.append('/nfs/zhengmeisong/wkspace/gesture/caffe/python')
import caffe
import numpy as np
import numpy.random as npr
import scipy.io as sio
import os
import cv2
import argparse
import time

if __name__ == '__main__':
    bbox_reg_net = caffe.Net("examples/hand_reg/lm87_64/test.prototxt", "models/fromAli/0927data_b512_iter_1000000.caffemodel", caffe.TEST)
#    bbox_reg_net = caffe.Net("/Users/momo/Desktop/sdk/momocv2_model/converted_model/hand_gesture/hand_gesture_reg_v3.0.prototxt", \
#                             "/Users/momo/Desktop/sdk/momocv2_model/converted_model/hand_gesture/hand_gesture_reg_v3.0.caffemodel", caffe.TEST)
    fid = open("data/1021data/Txts/T_onezanbigv_64S2030_16black.txt","r")
    TP=0
    inputSize = 64

    mean = 128
    lines = fid.readlines()
    fid.close()
    cur_=0
    sum_=len(lines)
    regloss = np.array([])
    probs = np.array([])
    roi_n = 0
    cls_n = 0
    totalTime = 0
    for line in lines:
        cur_+=1
        if not line:
            break;
        words = line.split()
        image_file_name = "/Users/momo/wkspace/caffe_space/detection/caffe/data/1021data/" + words[0]
        print cur_, image_file_name

        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
#        cv2.imshow("input",im)
#        cv2.waitKey()
        if h!=inputSize or w!=inputSize:
            im = cv2.resize(im,(int(inputSize),int(inputSize)))
        
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
        im  = im.astype(np.int)
#        print "before:",im.dtype
#        print im[0:4,0:4,0:4]
        im -= mean
#        print "after:",im.dtype
#        print im[0:4,0:4,0:4]
        label    = int(words[1])
        roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
        print "gt:", label, roi
        bbox_reg_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        bbox_reg_net.blobs['data'].data[...]=im

        startT48 = time.clock()
        out_ = bbox_reg_net.forward()
        endT48 = time.clock()
        totalTime += (endT48-startT48)

        if label != 0:
            roi_n+=1
        box_deltas = out_['fullyconnected1'][0]
        print box_deltas
        regloss = np.append(regloss,np.sum((box_deltas-roi)**2)/2)
        
    print "reg loss mean=", np.mean(regloss),"reg loss std=", np.std(regloss),"time:", totalTime*1000/cur_, "ms"
