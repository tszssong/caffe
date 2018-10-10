# -*- coding: utf-8 -*-
#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
    Demo script showing detections in sample images.
    
    See README.md for installation instructions before running.
"""
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys
# Make sure that caffe is on the python path:
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import argparse


if __name__ == "__main__":
    
    caffe.set_mode_cpu
#    net = caffe.Net("/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/models/pascal_voc/MMCV5_/faster_rcnn_end2end/test.prototxt",\
#                    "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/faster_rcnn_models/MMCV5_faster_rcnn_final.caffemodel", \
#                    caffe.TEST)
    net = caffe.Net("/Users/momo/Desktop/sdk/momocv2_model/converted_model/hand_gesture/hand_gesture_reg_v3.0.prototxt",\
                    "/Users/momo/Desktop/sdk/momocv2_model/converted_model/hand_gesture/hand_gesture_reg_v3.0.caffemode", \
                    caffe.TEST)
    #第一个卷基层的权值
    conv1_w = net.params['conv0'][0].data
    #第一个卷基层的偏置值
#    conv1_b = net.params['conv1_small'][1].data
    print "conv1_W:"
    print conv1_w,'\n'
#    print "conv1_b:"
#    print conv1_b,'\n'
    print conv1_w.size
    #第一个卷基层的权值
    conv2_w = net.params['mm_conv2'][0].data
    #第一个卷基层的偏置值
    conv2_b = net.params['mm_conv2'][1].data
    #可以打印相应的参数和参数的维度等信息
    print "conv2_W:"
    print conv2_w,'\n'
    print "conv2_b:"
    print conv2_b,'\n'
    print conv2_w.size,conv2_b.size
    net.save('/Users/momo/Desktop/out/rm_pred_box_layer.caffemodel')
