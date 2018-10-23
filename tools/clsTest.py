import os, sys
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/build/python')
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/python')
import numpy as np
import numpy.random as npr
#import scipy.io as sio
import caffe
import cv2
import time

NumTest = 100000
prototxt = "examples/hand_cls/mouth48/bnTestNew.prototxt"
#caffemodel = "models/mouth48_1012/1017f118w_iter_1100000.caffemodel"
caffemodel = "models/fromAli/mouth48bn/newdata_iter_350000.caffemodel"
#prototxt   = "no_bn.prototxt"
#caffemodel = "no_bn.caffemodel"
#prototxt   = "examples/hand_cls/mouth48/bnTest.prototxt"
#caffemodel = "models/mouth48_1012/1015bg_bn_iter_1200000.caffemodel"
#prototxt   = "examples/hand_cls/mouth48/bnTest.prototxt"
#caffemodel = "models/fromAli/1015bg_bn_iter_1765000.caffemodel"
#prototxt   = "examples/hand_cls/mouth48/bnTestOld.prototxt"
#caffemodel = "examples/hand_cls/mouth48/adbg_drop_iter_1180000.caffemodel"
#prototxt   = "examples/hand_cls/mouth48/test14cls.prototxt",
#caffemodel = "models/fromAli/1012_iter_2480000.caffemodel"
#prototxt   = "models/fromAli/mouth64bn/test.prototxt"
#caffemodel = "models/fromAli/mouth64bn/1018addbg_1012f_iter_1000000.caffemodel"
#caffemodel = "models/fromAli/mouth64bn/1018addbg_1012f0001_iter_1000000.caffemodel"

if __name__ == '__main__':
    
    caffe.set_mode_cpu()
    inputSize = 48
    mean = np.array([104, 117, 123])
    classify_net = caffe.Net(prototxt, caffemodel, caffe.TEST)
#    fid = open("data/48Test/Txts/5-five-wsTest_48R110S1020_1013_1.txt","r")
    fid = open("data/48Test/Txts/48test1020.txt","r")
#    fid = open("data/48Test/Txts/2cls_shuffle.txt","r")
#    fid = open("/Users/momo/Downloads/test0627.txt","r")
    subdirlists = ['bg', 'heart', 'yearh', 'one', 'baoquan', 'five', 'bainian', 'zan', 'fingerheart', 'ok', 'call', 'rock', 'big_v','fist']
    tp_dict = {}
    gt_dict = {}
    re_dict = {}
    for gname in subdirlists:
        tp_dict[gname] = 0
        gt_dict[gname] = 0
        re_dict[gname] = 0
    TP=0
    err = 0
    lines = fid.readlines()
    fid.close()
    cur_=0
    sum_=len(lines)
    regloss = np.array([])
    probs = np.array([])
    cls = 0
    totalTime = 0
    for line in lines:
        cur_+=1
        if not line or cur_ == NumTest:
            break;
        words = line.split()
#        image_file_name = "data/clsData/" + words[0]
#        image_file_name = "/Users/momo/Downloads/" + words[0]
        image_file_name = "data/48Test/" + words[0]

        if cur_%500 == 0:
            print cur_,
            sys.stdout.flush()
        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
        if h!=inputSize or w!=inputSize:
            im = cv2.resize(im,(int(inputSize),int(inputSize)))

        im = im.astype(np.int)
        im -= mean
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
        label    = int(words[1])
        if label > 13:
            continue
        gt_dict[subdirlists[label]] += 1

        classify_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        classify_net.blobs['data'].data[...]=im
        out_ = classify_net.forward()
        prob = out_['prob'][0]

        cls_prob = np.max(prob)
        cls = np.where(prob==np.max(prob))[0][0]
        re_dict[subdirlists[cls]] += 1
        if not cls == label:
            err += 1
        else:
            tp_dict[subdirlists[cls]]+=1
    print "\nerr, sum:",err, sum_
    reTotal = 0
    gtTotal = 0
    tpTotal = 0
    for gname in tp_dict:
        print "%12s"%(gname)," Recall:%.2f"%( float(tp_dict[gname])/float(gt_dict[gname]+1) ), \
              " Precision:%.2f"%( float(tp_dict[gname])/float(re_dict[gname]+1) ), \
              " re:%4d"%(re_dict[gname]), " tp:%4d"%(tp_dict[gname]), "gt:%4d"%(gt_dict[gname])
        
        reTotal += re_dict[gname]
        gtTotal += gt_dict[gname]
        tpTotal += tp_dict[gname]
    print "total recall:%.2f"%(float(tpTotal)/float(gtTotal)), "total precision:%.2f"%(float(tpTotal)/float(reTotal))
    print prototxt
    print caffemodel
