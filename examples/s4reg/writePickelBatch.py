import os, sys
import cv2
import numpy as np
import numpy.random as npr
import cPickle as pickle
wk_dir = "/Users/momo/wkspace/caffe_space/detection/caffe/data/1103reg64/"
InputSize = int(sys.argv[1])
BatchSize = int(sys.argv[2])
trainfile = "train.txt"
testfile = "test.txt"
print "gen imdb with for net input:", InputSize, "batchSize:", BatchSize

with open(wk_dir+trainfile, 'r') as f:
    trainlines = f.readlines()
with open(wk_dir+testfile, 'r') as f:
    testlines = f.readlines()
#######################################
# we seperate train data by batchsize #
#######################################
to_dir = wk_dir + "/trainIMDB/"
if not os.path.isdir(to_dir):
    os.makedirs(to_dir)

train_list = []
cur_ = 0
sum_ = len(trainlines)
for line in trainlines:
    cur_ += 1
    words = line.split()
    image_file_name = words[0]
    im = cv2.imread(wk_dir + image_file_name)
    h,w,ch = im.shape
    if h!=InputSize or w!=InputSize:
        im = cv2.resize(im,(InputSize,InputSize))
    roi = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    train_list.append([im, roi])
    if (cur_ % BatchSize == 0):
        print "write batch:" , cur_/BatchSize
        fid = open(to_dir +'train'+ str(BatchSize) + '_'+str(cur_/BatchSize),'w')
        pickle.dump(train_list, fid)
        fid.close()
        train_list[:] = []

print len(train_list), "train data generated\n"

###########################
# tests #
###########################
to_dir = wk_dir + "/testIMDB/"
if not os.path.isdir(to_dir):
    os.makedirs(to_dir)
test_list = []
cur_ = 0
sum_ = len(testlines)
for line in testlines:
   cur_ += 1
   words = line.split()
   image_file_name = words[0]
   im = cv2.imread(wk_dir + image_file_name)
   h,w,ch = im.shape
   if h!=InputSize or w!=InputSize:
       im = cv2.resize(im,(InputSize,InputSize))
   roi = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
   test_list.append([im, roi])

   if (cur_ % BatchSize == 0):
       print "write batch:", cur_ / BatchSize
       fid = open(to_dir +'test'+ str(BatchSize) + '_'+str(cur_/BatchSize), 'w')
       pickle.dump(test_list, fid)
       fid.close()
       test_list[:] = []
print len(test_list), "test data generated\n"
