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

mean = 128
with open(wk_dir +'./train.txt', 'r') as f:
    trainlines = f.readlines()
with open(wk_dir +'./test.txt', 'r') as f:
    testlines = f.readlines()

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()

train_list = []
cur_ = 0
sum_ = len(trainlines)
for line in trainlines:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = words[0]
    # print (image_file_name)
    im = cv2.imread(wk_dir + image_file_name)
    h,w,ch = im.shape
    if h!=InputSize or w!=InputSize:
        im = cv2.resize(im,(InputSize,InputSize))
    im  = np.swapaxes(im, 0, 2)
    im  = np.swapaxes(im, 1, 2)
    im  = im.astype(np.int)
    im -= mean
    roi = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    # print "words", words
    # print "roi", roi
    train_list.append([im, roi])

fid = open(wk_dir+"train_imdb",'w')
pickle.dump(train_list, fid)
fid.close()
print len(train_list), "train data generated"

test_list = []
cur_ = 0
sum_ = len(testlines)
for line in trainlines:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = words[0]
    # print (image_file_name)
    im = cv2.imread(wk_dir + image_file_name)
    h,w,ch = im.shape
    if h!=InputSize or w!=InputSize:
        im = cv2.resize(im,(InputSize,InputSize))
    im  = np.swapaxes(im, 0, 2)
    im  = np.swapaxes(im, 1, 2)
    im  = im.astype(np.int)
    im -= mean
    roi = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    # print "words", words
    # print "roi", roi
    test_list.append([im, roi])

fid = open(wk_dir+"test_imdb",'w')
pickle.dump(test_list, fid)
fid.close()
print len(test_list), "test data generated"
