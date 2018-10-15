import os, sys
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/build/python')
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/python')
import numpy as np
import numpy.random as npr
import scipy.io as sio
import caffe
import cv2
import time

NumTest = 20000
TH=0.7
if __name__ == '__main__':
    
    caffe.set_mode_cpu()
    inputSize = 128
    mean = np.array([104, 117, 123])

    classify_net = caffe.Net("/Users/momo/Desktop/gesture/fromAli/cls/cls_bareness.prototxt",
                            "/Users/momo/Desktop/gesture/fromAli/cls/ba_lmdb_b512_iter_40000.caffemodel", caffe.TEST)

    fid = open("/Users/momo/wkspace/Data/gesture/tests/128obj4test_0920RS13x10_5/pos_128obj4test_0920RS13x10_5.txt","r")
    subdirlists = ['bg', 'heart', 'yearh', 'one', 'baoquan', 'five', 'bainian', 'zan', 'fingerheart', 'ok', 'call', 'rock', 'big_v']
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
        image_file_name = "/Users/momo/wkspace/Data/gesture/tests/" + words[0]
#        print words, cur_

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
        gt_dict[subdirlists[label]] += 1

        classify_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        classify_net.blobs['data'].data[...]=im

        startT = time.clock()
        out_ = classify_net.forward()
        endT = time.clock()
        totalTime += (endT-startT)
        prob = out_['loss'][0]
#        print prob
#        cls = np.where(np.max(prob))
#        print np.max(prob), np.where(prob==np.max(prob))[0][0]
        cls_prob = np.max(prob)
        cls = np.where(prob==np.max(prob))[0][0]
        re_dict[subdirlists[cls]] += 1
        if not cls == label:
            err += 1
#            print words, subdirlists[label], cls, subdirlists[cls], cls_prob
        else:
            tp_dict[subdirlists[cls]]+=1
    print err, sum_
    print 'tp:', tp_dict
    print 're', re_dict
    print 'gt', gt_dict
    reTotal = 0
    gtTotal = 0
    tpTotal = 0
    for gname in tp_dict:
        print "%12s"%(gname)," recall:%.2f"%( float(tp_dict[gname])/float(gt_dict[gname]+1) ), " precision:%.2f"%( float(tp_dict[gname])/float(re_dict[gname]+1) )
        reTotal += re_dict[gname]
        gtTotal += gt_dict[gname]
        tpTotal += tp_dict[gname]
    print "total recall:%.2f"%(float(tpTotal)/float(gtTotal)), "total precision:%.2f"%(float(tpTotal)/float(reTotal))

#    print "TP/total:", float(TP)/float(cur_),TP, cur_,
#    print "reg loss mean=", np.mean(regloss),"reg loss std=", np.std(regloss),"time:", totalTime*1000/cur_, "ms"
#    print "porb mean=", np.mean(probs),"prob std=", np.std(probs)

