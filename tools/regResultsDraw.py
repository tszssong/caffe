# -*- coding: utf-8 -*-
#!/usr/bin/env python
import sys
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/build/python')
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/python')
import numpy as np
# import matplotlib.pyplot as plt
#from PIL import Image
import numpy.random as npr
import scipy.io as sio
import os
import caffe
import cv2
import argparse
import time
def clip_boxes(boxes, im_shape):
    boxes[0] = max(boxes[0], 0)
    boxes[1] = min(boxes[1], im_shape[0])
    boxes[2] = max(boxes[2], 0)
    boxes[3] = min(boxes[3], im_shape[1])
    return boxes
ScaleFacetors = np.array([1,1,1,1])
#ScaleFacetors = np.array([10,10,5,5])
def bbox_reg(boxes, deltas, nw, nh):
    deltas[:]/=ScaleFacetors[:]
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    w = boxes[2]-boxes[0]
    h = boxes[3]-boxes[1]
    ctrx = boxes[0] + 0.5*w
    ctry = boxes[1] + 0.5*h
    
    dw = deltas[2]
    dh = deltas[3]
    nw = np.exp(dw) * w
    nh = np.exp(dh) * h

    nctrx = ctrx + deltas[0]*float(nw)
    nctry = ctry + deltas[1]*float(nh)

    pred_boxes[0] = nctrx - 0.5*nw # x1
    pred_boxes[1] = nctry - 0.5*nh # y1
    pred_boxes[2] = nctrx + 0.5*nw # x2
    pred_boxes[3] = nctry + 0.5*nh # y2
    return pred_boxes

if __name__ == '__main__':
    bbox_reg_net = caffe.Net("/Users/momo/wkspace/caffe_space/caffe/models/prototxt/64fromali/test.prototxt" ,\
                             "/Users/momo/wkspace/caffe_space/caffe/models/prototxt/64fromali/0922allfb256_iter_510000.caffemodel", caffe.TEST)
    # bbox_reg_net = caffe.Net("/Users/momo/Desktop/sdk/momocv2_model/converted_model/hand_gesture/hand_gesture_reg_v3.0.prototxt", \
    #                         "/Users/momo/Desktop/sdk/momocv2_model/converted_model/hand_gesture/hand_gesture_reg_v3.0.caffemodel", caffe.TEST)
    # bbox_reg_net = caffe.Net("/Users/momo/wkspace/caffe_space/detection/caffe/examples/hand_reg/lm87_64/test.prototxt", \
    #                          "/Users/momo/wkspace/caffe_space/detection/caffe/models/lm87_64/0922allf_iter_200000.caffemodel", caffe.TEST)

    fid = open("/Users/momo/wkspace/caffe_space/detection/caffe/data/reg64Data/tests.txt","r")
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
        image_file_name = "/Users/momo/wkspace/caffe_space/detection/caffe/data/reg64Data/" + words[0]
        print cur_, image_file_name

        im = cv2.imread(image_file_name)
        print im.dtype
        h,w,ch = im.shape
        cv2.imshow("input",im)
        img=caffe.io.load_image(image_file_name)
        # plt.imshow(img)
        # plt.show()

        if h!=inputSize or w!=inputSize:
            im = cv2.resize(im,(int(inputSize),int(inputSize)))

        # im  = im.astype(np.int)
        # im -= np.array([[[104, 117, 123]]])
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
        inputTotal = im[0, :, :] * 256
        print im.shape
        for i in xrange(im.shape[0] - 1):
            inputTotal = np.hstack((inputTotal, im[i + 1, :, :] * 256))

        cv2.imshow("3 channel", inputTotal)

        print "im:\n",im[0,:,:][0:4, 0:4]

        # dst = np.zeros((im.shape[1], im.shape[2]))
        # dst = cv2.normalize(im[0,:,:], dst, 0, 255, cv2.NORM_MINMAX)
        # print "dst:\n",dst[0:4, 0:4]
        # cv2.imshow("i0",dst)
#        print "before:",im.dtype
#         print im[0:4,0:4,0:4]
        im = im.astype(np.int)
        im -= mean
        # imTotal = np.hstack((im[0,:,:]*256, im[1,:,:]*256, im[2,:,:]*256))
        imTotal = im[0,:,:]*256
        print im.shape
        for i in xrange( im.shape[0]-1 ):
            imTotal = np.hstack( (imTotal, im[i+1,:,:]*256) )
        # cv2.imshow("im0",im[0,:,:]*256)
        # cv2.imshow("im1",im[1,:,:]*256)
        # cv2.imshow("im2",im[2,:,:]*256)
        cv2.imshow("caffe input", imTotal)

        label    = int(words[1])
        roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
        print "gt:", label, roi
        bbox_reg_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        bbox_reg_net.blobs['data'].data[...]=im


        startT48 = time.clock()
        out_ = bbox_reg_net.forward()
        endT48 = time.clock()
        totalTime += (endT48-startT48)

        fea_smallconv0 = bbox_reg_net.blobs["conv0"].data[0]
        print "fea_smallconv0:\n", fea_smallconv0.shape, fea_smallconv0.dtype
        cv2.imshow("fea0", fea_smallconv0[0, :, :] * 255)
        fea0Total = fea_smallconv0[0, :, :] * 255
        for i in xrange(fea_smallconv0.shape[0] - 1):
            fea0Total = np.hstack((fea0Total, fea_smallconv0[i + 1, :, :] * 255))
        cv2.imshow("fea0", fea0Total)

        fea_plus0 = bbox_reg_net.blobs["_plus0"].data[0]
        print "fea_plus0:\n", fea_plus0.shape, fea_plus0.dtype
        cv2.imshow("_plus0", fea_plus0[0, :, :] * 255)
        fea_plus0Total = fea_plus0[0, :, :] * 255
        for i in xrange(fea_plus0.shape[0] - 1):
            fea_plus0Total = np.hstack((fea_plus0Total, fea_plus0[i + 1, :, :] * 255))
        cv2.imshow("_plus0", fea_plus0Total)

        if label != 0:
            roi_n+=1
        box_deltas = out_['fullyconnected1'][0]
        print box_deltas
        regloss = np.append(regloss,np.sum((box_deltas-roi)**2)/2)
        cv2.waitKey()
        
#    print "reg loss mean=", np.mean(regloss),"reg loss std=", np.std(regloss),"time:", totalTime*1000/cur_, "ms"
#    print "porb mean=", np.mean(probs),"prob std=", np.std(probs)
