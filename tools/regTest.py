# -*- coding: utf-8 -*-
#!/usr/bin/env python
import sys
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/build/python')
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/python')
import caffe
import numpy as np
import numpy.random as npr
import scipy.io as sio
import os
import cv2
import argparse
import time
#model = "examples/hand_reg/lm87_64/test.prototxt"
#model = "examples/hand_reg/mouthmac/test.prototxt"
#weights = "models/fromAli/mouth/1023f_addfisthebing_iter_1570000.caffemodel"
model = "/Users/momo/wkspace/caffe_space/detection/caffe/models/fromAli/mouth/test.prototxt";
weights = "/Users/momo/wkspace/caffe_space/detection/caffe/models/fromAli/mouth/25262730f600w_iter_990000.caffemodel";
if __name__ == '__main__':
    bbox_reg_net = caffe.Net( model, weights, caffe.TEST)
    fid = open("data/regTest/Txts/3-distrub_64S2430_5_3black.txt","r")
#    fid = open("data/regTest/test.txt","r")
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
        image_file_name = "/Users/momo/wkspace/caffe_space/detection/caffe/data/regTest/" + words[0]
        print cur_, image_file_name

        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
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
        
    print "reg loss mean =", np.mean(regloss)
    print "reg loss std  =", np.std(regloss)
    print "time:", totalTime*1000/cur_, "ms"
print model
print weights
