import sys
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/build/python')
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/python')
import cv2
import caffe
import numpy as np
import random
import cPickle as pickle
imdb_exit = False
###############################################################################
class Data_Layer_test(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 128
        net_side = 64
        roi_list = []
        roi_root = '/Users/momo/wkspace/caffe_space/detection/caffe/data/reg64Data/'
        roi_txt='/Users/momo/wkspace/caffe_space/detection/caffe/data/reg64Data/tests.txt'
        print roi_txt
        self.batch_loader = BatchLoader(roi_list,net_side,roi_root,roi_txt)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size, 4)
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        loss_task = 1
        for itt in range(self.batch_size):
            im, roi= self.batch_loader.load_next_image(loss_task)
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = roi
    def backward(self, top, propagate_down, bottom):
        pass
################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 128
        net_side = 64
        roi_list = ''
        roi_root = '/Users/momo/wkspace/caffe_space/detection/caffe/data/reg64Data/'
        roi_txt='/Users/momo/wkspace/caffe_space/detection/caffe/data/reg64Data/train1030s.txt'
        print roi_txt
        self.batch_loader = BatchLoader(roi_list,net_side,roi_root,roi_txt)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size, 4)
        
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        loss_task = 1
        for itt in range(self.batch_size):
            im, roi= self.batch_loader.load_next_image(loss_task)
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = roi
    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self,roi_list,net_side,roi_root,roi_txt):
        self.mean = 128
        self.im_shape = net_side
        self.roi_root = roi_root
        self.roi_list = []

        print "\nStart Reading:",roi_txt," into Memory...\n"
        fid = open(roi_txt,'r')
        lines = fid.readlines()
        fid.close()
        cur_=0
        sum_=len(lines)
        for line in lines:
            cur_+=1
            words = line.split()
            image_file_name = self.roi_root + words[0]
            roi= [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
            self.roi_list.append([image_file_name,roi])
        random.shuffle(self.roi_list)
        self.roi_cur = 0
        print "\n",str(len(self.roi_list))," Regression Data have been read into Memory...\n"

    def load_next_image(self,loss_task):
        if loss_task == 1:
            if self.roi_cur == len(self.roi_list):
                self.roi_cur = 0
                random.shuffle(self.roi_list)
            cur_data = self.roi_list[self.roi_cur]  # Get the image index
            image_file_name = cur_data[0]
            im = cv2.imread(image_file_name)
            h,w,ch = im.shape
            if h!=self.im_shape or w!=self.im_shape:
                im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
            im  = np.swapaxes(im, 0, 2)
            im  = np.swapaxes(im, 1, 2)
            im  = im.astype(np.int)
            im -= self.mean
            roi = cur_data[1]
            self.roi_cur += 1
            return im, roi
################################################################################
############SMOOTH L1 Regression Loss Layer By Python###########################
################################################################################
class smoothL1regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension")
        roi = bottom[1].data
        self.valid_index = np.where(roi[:,0] != -1)[0]
        self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self,bottom,top):
        self.diff[...] = 0
        top[0].data[...] = 0
        if self.N != 0:
            self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
            l_idx = np.where(self.diff<1)
            h_idx = np.where(self.diff>=1)
            top[0].data[...] = (np.sum(self.diff[l_idx]**2)/2. + np.sum( np.abs(self.diff[h_idx]) ) ) / bottom[0].num
 #       print "smooth l1:", top[0].data
    def backward(self,top,propagate_down,bottom):
        for i in range(2):
            if not propagate_down[i] or self.N==0:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
################################################################################
######################Regression Loss Layer By Python###########################
################################################################################
class regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	if bottom[0].count != bottom[1].count:
	    raise Exception("Input predict and groundTruth should have same dimension")
	roi = bottom[1].data
	self.valid_index = np.where(roi[:,0] != -1)[0]
	self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self,bottom,top):
	self.diff[...] = 0
	top[0].data[...] = 0
	if self.N != 0:
	    self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
            top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.
#        print "l2 loss:", top[0].data
    def backward(self,top,propagate_down,bottom):
	for i in range(2):
	    if not propagate_down[i] or self.N==0:
		continue
	    if i == 0:
		sign = 1
	    else:
		sign = -1
	    bottom[i].diff[...] = sign * self.diff / bottom[i].num
################################################################################
#############################Classify Layer By Python###########################
################################################################################
class cls_Layer_fc(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	top[0].reshape(len(bottom[1].data), 2,1,1)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]

class cls_Layer(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	top[0].reshape(len(bottom[1].data), 2)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]

