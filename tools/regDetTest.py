import os, sys
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/build/python')
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/python')
import numpy as np
import numpy.random as npr
#import scipy.io as sio
import caffe
import cv2
import time

prototxt = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/hand_reg/128mouth/128mouth/test.prototxt"
caffemodel = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/hand_reg/128mouth/4wtrain_iter_40000.caffemodel"

inputSize = 128
if __name__ == '__main__':
    
    caffe.set_mode_cpu()
    mean = 128
#    mean = np.array([128, 128, 128])
    det_net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    cap = cv2.VideoCapture(0)

while(1):
    ret,frame = cap.read()
    im = cv2.resize(frame, (int(inputSize),int(inputSize)) )
    display = im
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im  = im.astype(np.int)
    im -= mean
    
    det_net.blobs['data'].reshape(1,3,inputSize,inputSize)
    det_net.blobs['data'].data[...]=im
    out_ = det_net.forward()
    box = out_['fullyconnected1'][0]
    print box
    display = cv2.rectangle(display,(box[0],box[1]), (box[2],box[3]), (255, 0, 0),1)
    cv2.imshow("im",display)
    cv2.waitKey()

